use crate::SumcheckGrinding;
use p3_challenger::{FieldChallenger, GrindingChallenger};
use p3_field::{ExtensionField, TwoAdicField};
use rayon::prelude::*;
use utils::{Evaluation, univariate_selectors};
use whir_p3::{
    fiat_shamir::{errors::ProofError, verifier::VerifierState},
    poly::{dense::WhirDensePolynomial, evals::EvaluationsList},
};

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
    eq_factor: Option<&[EF]>,
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
        eq_factor,
        is_zerocheck,
        1,
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
    )
}

pub fn verify_with_univariate_skip<F, EF, Challenger>(
    eq_factor: Option<&[EF]>,
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
        eq_factor,
        is_zerocheck,
        skips,
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
    )
}

fn verify_core<EF, F, Challenger>(
    eq_factor: Option<&[EF]>,
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
    let mut round = 0;

    for (&deg, sumation_set) in max_degree_per_vars.iter().zip(sumation_sets) {
        let mut p_evals = Vec::<(F, EF)>::new();
        let mut eq_evals = Vec::<(F, EF)>::new();

        if is_zerocheck {
            if first_round {
                let proof_data = verifier_state
                    .next_extension_scalars_vec(deg + 2 - (1 << skips) - (1 << skips))?;
                p_evals.extend((0..(1 << skips)).map(|i| (F::from_usize(i), EF::ZERO)));
                p_evals.extend(
                    ((1 << skips)..=(deg + 2 - (1 << skips)))
                        .zip(proof_data)
                        .map(|(i, eval)| (F::from_usize(i), eval))
                        .collect::<Vec<_>>(),
                );

                if let Some(eq_factor) = &eq_factor {
                    let selectors = univariate_selectors::<F>(skips);
                    eq_evals = (0..1 << skips)
                        .into_par_iter()
                        .map(|i| (F::from_usize(i), selectors[i].evaluate(eq_factor[round])))
                        .collect::<Vec<_>>();
                }
            } else {
                let proof_data =
                    verifier_state.next_extension_scalars_vec(deg + 2 - (1 << skips))?;
                p_evals.extend(
                    (0..(deg + 2 - (1 << skips)))
                        .zip(proof_data)
                        .map(|(i, eval)| (F::from_usize(i), eval))
                        .collect::<Vec<_>>(),
                );

                if let Some(eq_factor) = &eq_factor {
                    let selectors = univariate_selectors::<F>(skips);
                    eq_evals = (0..1 << skips)
                        .into_par_iter()
                        .map(|i| (F::from_usize(i), selectors[i].evaluate(eq_factor[round])))
                        .collect::<Vec<_>>();
                }
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

        // let coeffs = verifier_state.next_extension_scalars_vec(deg + 1)?;
        // let pol = WhirDensePolynomial::from_coefficients_vec(coeffs);

        let computed_sum = sumation_set.iter().map(|&s| pol.evaluate(s)).sum();

        if first_round {
            first_round = false;
            sum = computed_sum;
        } else if target != computed_sum {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = verifier_state.sample();

        let pow_bits = grinding.pow_bits::<EF>(deg);
        verifier_state.check_pow_grinding(pow_bits)?;

        target = pol.evaluate(challenge);
        challenges.push(challenge);
        skips = 1;
        round += 1;
    }

    Ok((
        sum,
        Evaluation {
            point: challenges,
            value: target,
        },
    ))
}
