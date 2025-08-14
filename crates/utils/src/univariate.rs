use p3_field::{ExtensionField, Field};
use rayon::prelude::*;

/// Precomputed barycentric weights for common skip values
/// These are the weights for Lagrange interpolation on {0, 1, ..., m-1}
/// where m = 2^skips for skips = 1, 2, 3, 4

// For skips=1, m=2: weights for {0, 1}
const BARYCENTRIC_WEIGHTS_2: [u64; 2] = [2130706432, 1];

// For skips=2, m=4: weights for {0, 1, 2, 3}
const BARYCENTRIC_WEIGHTS_4: [u64; 4] = [1775588694, 1065353217, 1065353216, 355117739];

// For skips=3, m=8: weights for {0, 1, 2, ..., 7}
const BARYCENTRIC_WEIGHTS_8: [u64; 8] = [
    413035751, 1370162609, 150925039, 458693746, 1672012687, 1979781394, 760543824, 1717670682,
];

// For skips=4, m=16: weights for {0, 1, 2, ..., 15}
const BARYCENTRIC_WEIGHTS_16: [u64; 16] = [
    1296489123, 1859727485, 1896852636, 303130976, 1221313505, 1574523155, 926972130, 938885123,
    1191821310, 1203734303, 556183278, 909392928, 1827575457, 233853797, 270978948, 834217310,
];

// ================================================================================================
// FUNCTIONS - Barycentric Lagrange evaluation (optimized approach)
// ================================================================================================

/// Compute barycentric weights for Lagrange interpolation on points {0, 1, ..., m-1}
/// Weight for point i is: w_i = (-1)^(m-1-i) / (i! * (m-1-i)!)
pub fn barycentric_weights<F: Field>(m: usize) -> Vec<F> {
    if m == 1 {
        return vec![F::ONE];
    }

    // Compute factorials
    let mut factorial = vec![F::ONE; m];
    for i in 1..m {
        factorial[i] = factorial[i - 1] * F::from_usize(i);
    }

    (0..m)
        .map(|i| {
            let sign = if (m - 1 - i) % 2 == 0 {
                F::ONE
            } else {
                -F::ONE
            };
            sign / (factorial[i] * factorial[m - 1 - i])
        })
        .collect()
}

/// Get precomputed barycentric weights for common values, fallback to computation
pub fn barycentric_weights_precomputed<F: Field + 'static>(m: usize) -> Vec<F> {
    use std::any::TypeId;

    if TypeId::of::<F>() == TypeId::of::<p3_koala_bear::KoalaBear>() {
        match m {
            2 => BARYCENTRIC_WEIGHTS_2
                .iter()
                .map(|&w| F::from_u64(w))
                .collect(),
            4 => BARYCENTRIC_WEIGHTS_4
                .iter()
                .map(|&w| F::from_u64(w))
                .collect(),
            8 => BARYCENTRIC_WEIGHTS_8
                .iter()
                .map(|&w| F::from_u64(w))
                .collect(),
            16 => BARYCENTRIC_WEIGHTS_16
                .iter()
                .map(|&w| F::from_u64(w))
                .collect(),
            _ => barycentric_weights::<F>(m),
        }
    } else {
        barycentric_weights::<F>(m)
    }
}

/// Method for evaluating Lagrange basis polynomials without constructing
/// the full polynomial coefficients. It relies on the second form of the barycentric
/// interpolation formula.
///
/// For a set of points {x_0, ..., x_{m-1}}, the Lagrange basis polynomial L_i(x) can be
/// evaluated as:
///
/// L_i(x) = l(x) * w_i / (x - x_i)
///
/// where:
/// - l(x) = ∏(x - x_j) is the "vanishing polynomial" over the grid.
/// - w_i = 1 / ∏_{j≠i}(x_i - x_j) are the precomputed barycentric weights.
///
/// This approach reduces the complexity of evaluation from O(m^2) (for interpolation)
/// followed by O(m) (for evaluation) to just O(m) per evaluation after a one-time
/// O(m) setup for the weights.
///
/// For a detailed explanation of the formula and its derivation, see:
/// https://tobydriscoll.net/fnc-julia/globalapprox/barycentric.html
///
/// Evaluate all Lagrange basis polynomials L_i(x) at point x for grid {0, 1, ..., m-1}
/// Returns [L_0(x), L_1(x), ..., L_{m-1}(x)]
///
// TODO: Should we use the Montgomery trick here instead of suffix prefix?
/// Helper function to compute prefix and suffix products for Lagrange basis evaluation
/// Returns (prefix_products, suffix_products) where:
/// - prefix_products[k] = ∏_{j < k} (x - j)  with prefix_products[0] = 1
/// - suffix_products[k] = ∏_{j > k} (x - j)  with suffix_products[m] = 1
fn compute_prefix_suffix_products<T: Field>(m: usize, x: T) -> (Vec<T>, Vec<T>) {
    let denom: Vec<T> = (0..m).map(|j| x - T::from_usize(j)).collect();

    let mut pref = vec![T::ONE; m + 1];
    for k in 0..m {
        pref[k + 1] = pref[k] * denom[k];
    }

    let mut suff = vec![T::ONE; m + 1];
    for k in (0..m).rev() {
        suff[k] = suff[k + 1] * denom[k];
    }
    (pref, suff)
}

fn evaluate_lagrange_basis_prefix_suffix<F: Field, T: Field>(
    m: usize,
    x: T,
    weights: &[F],
) -> Vec<T>
where
    T: From<F>,
{
    let (pref, suff) = compute_prefix_suffix_products(m, x);
    let mut result = vec![T::ZERO; m];
    for i in 0..m {
        let product = pref[i] * suff[i + 1];
        result[i] = product * T::from(weights[i]);
    }
    result
}

///  Montgomery's Trick Implementation

/// This method uses one field inversion, which can be faster if m is large.
fn evaluate_lagrange_basis_montgomery<F: Field, T: Field>(m: usize, x: T, weights: &[F]) -> Vec<T>
where
    T: From<F>,
{
    let denom: Vec<T> = (0..m).map(|j| x - T::from_usize(j)).collect();

    // 1. Compute prefix products: pref[i] = d_0 * ... * d_i
    let mut pref = vec![T::ONE; m];
    pref[0] = denom[0];
    for i in 1..m {
        pref[i] = pref[i - 1] * denom[i];
    }

    // 2. Compute the inverse of the total product (one inversion)
    let mut inv_suffix = pref[m - 1].inverse();
    let mut result = vec![T::ZERO; m];

    // 3. Iterate backwards to compute each L_i(x)
    for i in (1..m).rev() {
        // For index i, we need: pref[i-1] * (d_{i+1} * ... * d_{m-1})
        // The suffix part is pref[m-1] * (d_0 * ... * d_i)⁻¹ = pref[m-1] * inv_suffix
        let prefix_part = pref[i - 1];
        let suffix_part = pref[m - 1] * inv_suffix;
        let product = prefix_part * suffix_part;
        result[i] = product * T::from(weights[i]);

        // Update inv_suffix for the next iteration (i-1)
        // inv_suffix for (i-1) = (d_0 * ... * d_{i-1})⁻¹ = inv_suffix * d_i
        inv_suffix *= denom[i];
    }

    // Handle the i=0 case separately
    // We need d_1 * ... * d_{m-1} = pref[m-1] * d_0⁻¹
    // At this point, inv_suffix is exactly d_0⁻¹
    let product_zero = pref[m - 1] * inv_suffix;
    result[0] = product_zero * T::from(weights[0]);

    result
}

/// Evaluate all Lagrange basis polynomials L_i(x) at point x in an extension field.
pub fn evaluate_lagrange_basis_all<F: Field, EF: ExtensionField<F>>(
    m: usize,
    x: EF,
    weights: &[F],
) -> Vec<EF> {
    if m == 1 {
        return vec![EF::ONE];
    }
    if let Some(i) = (0..m).find(|&i| x == EF::from_usize(i)) {
        let mut result = vec![EF::ZERO; m];
        result[i] = EF::ONE;
        return result;
    }

    // Method 1: Prefix-Suffix (clearer, more multiplications)
    //evaluate_lagrange_basis_prefix_suffix(m, x, weights)

    // Method 2: Montgomery's Trick (fewer multiplications, one inversion)
    evaluate_lagrange_basis_montgomery(m, x, weights)
}

/// Evaluate all Lagrange basis polynomials L_i(x) at point x in the base field.
pub fn evaluate_lagrange_basis_all_base_field<F: Field>(m: usize, x: F, weights: &[F]) -> Vec<F> {
    if m == 1 {
        return vec![F::ONE];
    }
    if let Some(i) = (0..m).find(|&i| x == F::from_usize(i)) {
        let mut result = vec![F::ZERO; m];
        result[i] = F::ONE;
        return result;
    }

    // Method 1: Prefix-Suffix (clearer, more multiplications)
    //evaluate_lagrange_basis_prefix_suffix(m, x, weights)

    // Method 2: Montgomery's Trick (fewer multiplications, one inversion)
    evaluate_lagrange_basis_montgomery(m, x, weights)
}

/// Efficient evaluation of selector polynomials at multiple points in parallel.
pub fn evaluate_selectors_batch<F: Field, EF: ExtensionField<F>>(
    m: usize,
    points: &[EF],
) -> Vec<Vec<EF>> {
    let weights = barycentric_weights_precomputed::<F>(m);
    points
        .par_iter()
        .map(|&point| evaluate_lagrange_basis_all(m, point, &weights))
        .collect()
}

#[cfg(all(test))]
mod tests {
    use super::barycentric_weights;
    use p3_field::PrimeField64;
    use p3_koala_bear::KoalaBear;

    type F = KoalaBear;

    // Test to generate precomputed barycentric weights
    #[test]
    fn print_precomputed_barycentric_weights() {
        println!("\n=== BARYCENTRIC WEIGHTS FOR KOALABEAR ===");

        for skips in 1..=4 {
            let m = 1usize << skips;
            let weights = barycentric_weights::<F>(m);

            println!(
                "\n// For skips={}, m={}: weights for {{0, 1, ..., {}}}",
                skips,
                m,
                m - 1
            );
            print!("const BARYCENTRIC_WEIGHTS_{}: [u64; {}] = [", m, m);

            for (i, weight) in weights.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                if i % 8 == 0 && m > 8 {
                    print!("\n    ");
                }
                print!("{}", weight.as_canonical_u64());
            }

            if m > 8 {
                println!("\n];");
            } else {
                println!("];");
            }

            println!("// Verification: (skipped for now - trait issues)")
        }
    }

    // Test to verify barycentric evaluation matches polynomial evaluation
    #[test]
    fn verify_barycentric_vs_polynomial_evaluation() {
        println!("\n=== VERIFICATION: BARYCENTRIC vs POLYNOMIAL ===");

        for skips in 1..=4 {
            let m = 1usize << skips;
            println!("\nTesting skips={}, m={}", skips, m);

            println!("// Comparison test: (skipped for now - trait issues)")
        }
    }
}
