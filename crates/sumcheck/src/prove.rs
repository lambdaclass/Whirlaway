use std::{any::TypeId, borrow::Borrow};

use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{BasedVectorSpace, PackedValue};
use p3_field::{ExtensionField, Field, TwoAdicField};
use rayon::prelude::*;
use tracing::instrument;
use utils::{
    batch_fold_multilinear_in_large_field, batch_fold_multilinear_in_small_field,
    univariate_selectors,
};
use whir_p3::{
    fiat_shamir::prover::ProverState,
    poly::{dense::WhirDensePolynomial, evals::EvaluationsList},
};

use crate::{SumcheckComputation, SumcheckComputationPacked, SumcheckGrinding};

pub const MIN_VARS_FOR_GPU: usize = 0; // When there are a small number of variables, it's not worth using GPU

#[allow(clippy::too_many_arguments)]
pub fn prove<F, NF, EF, M, SC, Challenger>(
    skips: usize, // skips == 1: classic sumcheck. skips >= 2: sumcheck with univariate skips (eprint 2024/108)
    multilinears: &[M],
    computation: &SC,
    constraints_degree: usize,
    batching_scalars: &[EF],
    eq_factor: Option<&[EF]>,
    is_zerofier: bool,
    fs_prover: &mut ProverState<F, EF, Challenger>,
    mut sum: EF,
    n_rounds: Option<usize>,
    grinding: SumcheckGrinding,
    mut missing_mul_factor: Option<EF>,
) -> (Vec<EF>, Vec<EvaluationsList<EF>>, EF)
where
    F: TwoAdicField,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> + TwoAdicField,
    M: Borrow<EvaluationsList<NF>>,
    SC: SumcheckComputation<F, NF, EF>
        + SumcheckComputation<F, EF, EF>
        + SumcheckComputationPacked<F, EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let multilinears = multilinears.iter().map(|m| m.borrow()).collect::<Vec<_>>();
    let mut n_vars = multilinears[0].num_variables();
    assert!(multilinears.iter().all(|m| m.num_variables() == n_vars));

    let mut challenges = Vec::new();
    let n_rounds = n_rounds.unwrap_or(n_vars - skips + 1);
    if let Some(eq_factor) = &eq_factor {
        assert_eq!(eq_factor.len(), n_vars - skips + 1);
    }

    let mut folded_multilinears = sc_round(
        skips,
        &multilinears,
        &mut n_vars,
        computation,
        eq_factor,
        batching_scalars,
        is_zerofier,
        fs_prover,
        constraints_degree,
        &mut sum,
        grinding,
        &mut challenges,
        0,
        &mut missing_mul_factor,
    );

    for i in 1..n_rounds {
        folded_multilinears = sc_round(
            1,
            &folded_multilinears.iter().collect::<Vec<_>>(),
            &mut n_vars,
            computation,
            eq_factor,
            batching_scalars,
            false,
            fs_prover,
            constraints_degree,
            &mut sum,
            grinding,
            &mut challenges,
            i,
            &mut missing_mul_factor,
        );
    }

    (challenges, folded_multilinears, sum)
}

// This function applies a sumcheck round, building the round univariate polynomial `p` and returns the
// multilinear polynomials needed for the next round.
// For the documentation, we follow the notation in
// https://github.com/TomWambsgans/Whirlaway/blob/master/Whirlaway.pdf
#[instrument(name = "sumcheck_round", skip_all, fields(round))]
#[allow(clippy::too_many_arguments)]
pub fn sc_round<F, NF, EF, SC, Challenger>(
    skips: usize, // the first round will fold 2^skips (instead of 2 in the basic sumcheck)
    multilinears: &[&EvaluationsList<NF>], // `c^up` and `c^down` for each column c.
    n_vars: &mut usize,
    computation: &SC,         // In the zerocheck: constraints `h_i`.
    eq_factor: Option<&[EF]>, // In the zerocheck: the random vector `(r_0, ..., r_{n-1})`.
    batching_scalars: &[EF],
    is_zerofier: bool,
    fs_prover: &mut ProverState<F, EF, Challenger>,
    comp_degree: usize, // Maximum constraint degree. In poseidon2 it's 3.
    sum: &mut EF,
    grinding: SumcheckGrinding,
    challenges: &mut Vec<EF>,
    round: usize,
    missing_mul_factor: &mut Option<EF>,
) -> Vec<EvaluationsList<EF>>
where
    F: TwoAdicField,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F> + TwoAdicField,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    // The multilinear polynomial `eq(X, r)`.
    // In round `j` of the sumcheck eq_mle = eq((x_{j+1}, ... ,x_{n-1}), (r_{j+1}, ... , r_{n-1}))
    let eq_mle = eq_factor.map(|eq_factor| EvaluationsList::eval_eq(&eq_factor[1 + round..]));

    // `selectors` is a vector of 2^skips univariate polynomials.
    //
    // Example: If skips = 1, `selectors` has two polynomials S_0 and S_1.
    // S_0 is the interpolating polynomial of (0,1) and (1,0). I.e. S_0(x) = 1 - x
    // S_1 is the interpolating polynomial of (0,0) and (1,1). I.e. S_1(x) = x
    //
    // Example: If skips = 2, `selectors` has four polynomials S_0, S_1, S_2, S_3.
    // S_0 is the interpolating polynomial of (0,1), (1,0), (2,0), (3,0).
    // S_1 is the interpolating polynomial of (0,0), (1,1), (2,0), (3,0).
    // S_2 is the interpolating polynomial of (0,0), (1,0), (2,1), (3,0).
    // S_3 is the interpolating polynomial of (0,0), (1,0), (2,0), (3,1).
    let selectors = univariate_selectors::<F>(skips);

    // `p_evals` will collect the the interpolating points for the sumcheck round polynomial `p`.
    let mut p_evals = Vec::<(F, EF)>::new();
    // If it's a zerocehck, we start interpolating at the value 2^skips, since we know that in the previous values `p` is zero.
    let start = if is_zerofier {
        p_evals.extend((0..1 << skips).map(|i| (F::from_usize(i), EF::ZERO)));
        1 << skips
    } else {
        0
    };

    // for every value `z` we want to evaluate `p(z)`.
    for z in start..=comp_degree * ((1 << skips) - 1) {
        // If z == 2^skips - 1, we don't need to do the evaluation, we can deduce p(z) from the previous evaluations.
        //
        // Example: If skips = 1, we use that p(0) + p(1) = sum, then p(1) = sum - p(0).
        let sum_z = if z == (1 << skips) - 1 {
            if let Some(eq_factor) = eq_factor {
                (*sum
                    - (0..(1 << skips) - 1)
                        .map(|i| p_evals[i].1 * selectors[i].evaluate(eq_factor[round]))
                        .sum::<EF>())
                    / selectors[(1 << skips) - 1].evaluate(eq_factor[round])
            } else {
                *sum - p_evals.iter().map(|(_, s)| *s).sum::<EF>()
            }
        } else {
            // We evalaute each selector in z.
            // Example: If skips = 1, folding_scalars has 1 - z and z.
            let folding_scalars = selectors
                .iter()
                .map(|s| s.evaluate(F::from_usize(z)))
                .collect::<Vec<_>>();
            // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
            // `folded` has all the polynomials in `multilinears`, but each of them with the first `k` variables fixed, where k = skips.
            let folded = batch_fold_multilinear_in_small_field(multilinears, &folding_scalars);
            // We sum over the hypercube.
            let mut sum_z =
                compute_over_hypercube(&folded, computation, batching_scalars, eq_mle.as_ref());
            // A factor that we don't include in the previous sum and take out as common factor.
            // `missing_mul_factor` starts being `None`` at round 0.
            if let Some(missing_mul_factor) = missing_mul_factor {
                sum_z *= *missing_mul_factor;
            }

            sum_z
        };

        p_evals.push((F::from_usize(z), sum_z));
    }

    let mut p: WhirDensePolynomial<EF> =
        WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();

    if let Some(eq_factor) = &eq_factor {
        // https://eprint.iacr.org/2024/108.pdf Section 3.2
        // We do not take advantage of this trick to send less data, but we could do so in the future (TODO)
        // We multiply p by a polynomial given by the missing factor.
        // This polynomial is calculated by interpolating the evaluation of each selector in r_j, where j is the number of round.
        //
        // Example: If skips = 1, we interpolate (0, 1 - r_j) and (1, r_j).
        p *= &WhirDensePolynomial::lagrange_interpolation(
            &(0..1 << skips)
                .into_par_iter()
                .map(|i| (F::from_usize(i), selectors[i].evaluate(eq_factor[round])))
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }

    // We add the coefficients of p to the transcript.
    fs_prover.add_extension_scalars(&p.coeffs);

    // We prepare the parameters for the next round.
    let challenge = fs_prover.sample();
    challenges.push(challenge);
    *sum = p.evaluate(challenge);
    *n_vars -= skips;

    let pow_bits = grinding
        .pow_bits::<EF>((comp_degree + usize::from(eq_factor.is_some())) * ((1 << skips) - 1));
    fs_prover.pow_grinding(pow_bits);

    // We evaluate the selectors in the challenge.
    let folding_scalars = selectors
        .iter()
        .map(|s| s.evaluate(challenge))
        .collect::<Vec<_>>();
    if let Some(eq_factor) = eq_factor {
        // We update the missing_mul_challenge.
        //
        // Example: skips = 1, round 0.
        // `missing_mul_factor` = (1-s_0)(1-r_0) + s_0 * r_0
        //
        // Example: skips = 1, round 1.
        // `missing_mul_factor` = ((1-s_1)(1-r_1) + s_1 * r_1) * ((1-s_0)(1-r_0) + s_0 * r_0])
        *missing_mul_factor = Some(
            selectors
                .iter()
                .map(|s| s.evaluate(eq_factor[round]) * s.evaluate(challenge))
                .sum::<EF>()
                * missing_mul_factor.unwrap_or(EF::ONE),
        );
    }
    // If skips == 1 (ie classic sumcheck round, we could avoid 1 multiplication below: TODO not urgent)
    // We calculate the new multilinear polynomials for the next round by fixing the first variables using the folding scalars
    batch_fold_multilinear_in_large_field(multilinears, &folding_scalars)
}

fn compute_over_hypercube<F, NF, EF, SC>(
    pols: &[EvaluationsList<NF>],
    computation: &SC,
    batching_scalars: &[EF],
    eq_mle: Option<&EvaluationsList<EF>>,
) -> EF
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF> + ExtensionField<F>,
    SC: SumcheckComputation<F, NF, EF> + SumcheckComputationPacked<F, EF>,
{
    assert!(
        pols.iter()
            .all(|p| p.num_variables() == pols[0].num_variables())
    );
    let n_vars = pols[0].num_variables();
    if TypeId::of::<NF>() == TypeId::of::<F>() {
        let pols: &[EvaluationsList<F>] = unsafe { std::mem::transmute(pols) };
        let packed_pols = pols
            .iter()
            .map(|p| F::Packing::pack_slice(p.evals()))
            .collect::<Vec<_>>();

        let decomposed_batching_scalars: Vec<_> = (0..<EF as BasedVectorSpace<F>>::DIMENSION)
            .map(|i| {
                batching_scalars
                    .iter()
                    .map(|x| x.as_basis_coefficients_slice()[i])
                    .collect()
            })
            .collect();

        (0..(1 << n_vars) / F::Packing::WIDTH)
            .into_par_iter()
            .enumerate()
            .map(|(x, i)| {
                let point = packed_pols.iter().map(|pol| pol[x]).collect::<Vec<_>>();
                let res =
                    computation.eval_packed(&point, batching_scalars, &decomposed_batching_scalars);
                if let Some(eq_mle) = eq_mle {
                    res.enumerate()
                        .map(|(idx_in_packing, res)| {
                            res * eq_mle.evals()[i * F::Packing::WIDTH + idx_in_packing]
                        })
                        .sum()
                } else {
                    res.sum()
                }
            })
            .sum()
    } else {
        // TODO packing everywhere
        assert_eq!(TypeId::of::<NF>(), TypeId::of::<EF>());
        // For each `b` in the n-hypercube, we calculate the polynomial evaluations and the sum.
        (0..1 << n_vars)
            .into_par_iter()
            .map(|x| {
                // Example: In the zerocheck,
                // point = [(c_0)^{up} (b), ..., (c_{M-1})^{up} (b), (c_0)^{down} (b), ..., (c_{M-1})^{down} (b)]
                // `point` has 2 * M elements, where M is the number of columns.
                let point = pols.iter().map(|pol| pol.evals()[x]).collect::<Vec<_>>();
                let eq_mle_eval = eq_mle.map(|p| p.evals()[x]);
                eval_sumcheck_computation(computation, batching_scalars, &point, eq_mle_eval)
            })
            .sum()
    }
}

pub fn eval_sumcheck_computation<F, NF, EF, SC>(
    computation: &SC,
    batching_scalars: &[EF],
    point: &[NF],
    eq_mle_eval: Option<EF>,
) -> EF
where
    F: Field,
    NF: ExtensionField<F>,
    EF: ExtensionField<NF>,
    SC: SumcheckComputation<F, NF, EF>,
{
    let res = computation.eval(point, batching_scalars);
    eq_mle_eval.map_or(res, |factor| res * factor)
}
