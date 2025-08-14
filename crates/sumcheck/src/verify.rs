use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use utils::Evaluation;
use whir_p3::{
    fiat_shamir::{errors::ProofError, verifier::VerifierState},
    poly::dense::WhirDensePolynomial,
};

use crate::SumcheckGrinding;

#[derive(Debug, Clone)]
pub enum SumcheckError {
    Fs(ProofError),
    InvalidRound,
}

impl From<ProofError> for SumcheckError {
    fn from(e: ProofError) -> Self {
        Self::Fs(e)
    }
}

pub fn verify<F, EF, Challenger>(
    is_zerocheck: bool,
    verifier_state: &mut VerifierState<F, EF, Challenger>,
    n_vars: usize,
    degree: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let sumation_sets = vec![(0..2).map(|i| EF::from_usize(i)).collect::<Vec<_>>(); n_vars];
    let max_degree_per_vars = vec![degree; n_vars];
    verify_core(
        is_zerocheck,
        1,
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
    )
}

pub fn verify_with_univariate_skip<F, EF, Challenger>(
    is_zerocheck: bool,
    verifier_state: &mut VerifierState<F, EF, Challenger>,
    degree: usize,
    n_vars: usize,
    skips: usize,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    let mut max_degree_per_vars = vec![degree * ((1 << skips) - 1)];
    max_degree_per_vars.extend(vec![degree; n_vars - skips]);
    let mut sumation_sets = vec![(0..1 << skips).map(EF::from_usize).collect::<Vec<_>>()];
    sumation_sets.extend(vec![
        (0..2).map(EF::from_usize).collect::<Vec<_>>();
        n_vars - skips
    ]);
    let mut skips = skips;
    verify_core(
        is_zerocheck,
        skips,
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
    )
}

fn verify_core<EF, F, Challenger>(
    mut is_zerocheck: bool,
    mut skips: usize,
    verifier_state: &mut VerifierState<F, EF, Challenger>,
    max_degree_per_vars: &[usize],
    sumation_sets: Vec<Vec<EF>>,
    grinding: SumcheckGrinding,
) -> Result<(EF, Evaluation<EF>), SumcheckError>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>,
{
    assert_eq!(max_degree_per_vars.len(), sumation_sets.len(),);
    let mut challenges = Vec::new();
    let mut first_round = true;
    let (mut sum, mut target) = (EF::ZERO, EF::ZERO);

    for (&deg, sumation_set) in max_degree_per_vars.iter().zip(sumation_sets) {
        // println!("---ROUND---");

        let mut p_evals = Vec::<(F, EF)>::new();
        let mut eq_evals = Vec::<(F, EF)>::new();

        if is_zerocheck {
            if first_round {
                let proof_data =
                    verifier_state.next_extension_scalars_vec(deg + 2 - (1 << skips))?; // 2 para data_p_evals + 2 para data_eq_evals
                let data_p_evals = proof_data[..(deg + 2 - (1 << skips)) - (1 << skips)].to_vec();
                let data_eq_evals = proof_data[(deg + 2 - (1 << skips)) - (1 << skips)..].to_vec();
                p_evals.extend((0..(1 << skips)).map(|i| (F::from_usize(i), EF::ZERO)));
                p_evals.extend(
                    ((1 << skips)..=(deg + 2 - (1 << skips)))
                        .zip(data_p_evals)
                        .map(|(i, eval)| (F::from_usize(i), eval))
                        .collect::<Vec<_>>(),
                );
                eq_evals.extend(
                    (0..1 << skips)
                        .zip(data_eq_evals)
                        .map(|(i, eval)| (F::from_usize(i), eval))
                        .collect::<Vec<_>>(),
                );
            } else {
                let proof_data = verifier_state.next_extension_scalars_vec(deg + 2)?;
                let data_p_evals = proof_data[..deg + 2 - (1 << skips)].to_vec();
                let data_eq_evals = proof_data[deg + 2 - (1 << skips)..].to_vec();
                p_evals.extend(
                    (0..=deg)
                        .zip(data_p_evals)
                        .map(|(i, eval)| (F::from_usize(i), eval))
                        .collect::<Vec<_>>(),
                );
                eq_evals.extend(
                    (0..1 << skips)
                        .zip(data_eq_evals)
                        .map(|(i, eval)| (F::from_usize(i), eval))
                        .collect::<Vec<_>>(),
                );
            }
        } else {
            let proof_data = verifier_state.next_extension_scalars_vec(deg + 1)?;
            p_evals.extend(
                (0..=deg)
                    .zip(proof_data)
                    .map(|(i, eval)| (F::from_usize(i), eval))
                    .collect::<Vec<_>>(),
            );
        }
        let mut pol = WhirDensePolynomial::lagrange_interpolation(&p_evals).unwrap();
        if is_zerocheck {
            let eq_poly = WhirDensePolynomial::lagrange_interpolation(&eq_evals).unwrap();
            pol *= &eq_poly;
        }

        // println!("p_evals len: {:?}", p_evals.len());
        // println!("p_evals: {:?}", p_evals);
        // println!("eq_evals len: {:?}", eq_evals.len());
        // println!("eq_evals: {:?}", eq_evals);

        // let coeffs = verifier_state.next_extension_scalars_vec(deg + 1)?;
        // let pol = WhirDensePolynomial::from_coefficients_vec(coeffs);

        // println!("Degree: {deg}");
        // println!("Polynomial coeffs: {:?}", pol.coeffs);
        // println!("Polynomial coeff len: {:?}", pol.coeffs.len());
        // println!("sumation set: {:?}", sumation_set);

        // let pol_0 = pol.evaluate(EF::ZERO);
        // let pol_1 = pol.evaluate(EF::ONE);
        // println!("Polynomial at 0: {pol_0}");
        // println!("Polynomial at 1: {pol_1}");

        let computed_sum = sumation_set.iter().map(|&s| pol.evaluate(s)).sum();

        // println!("Computed sum: {computed_sum}");
        // println!("Target: {target}");

        if first_round {
            first_round = false;
            sum = computed_sum;
        } else if target != computed_sum {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = verifier_state.sample();
        //println!("Challenge: {challenge}");

        let pow_bits = grinding.pow_bits::<EF>(deg);
        verifier_state.check_pow_grinding(pow_bits)?;

        target = pol.evaluate(challenge);
        challenges.push(challenge);
        skips = 1;
    }

    Ok((
        sum,
        Evaluation {
            point: challenges,
            value: target,
        },
    ))
}
