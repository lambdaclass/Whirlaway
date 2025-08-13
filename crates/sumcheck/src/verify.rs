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
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
        false,
    )
}

pub fn verify_with_univariate_skip<F, EF, Challenger>(
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
    verify_core(
        verifier_state,
        &max_degree_per_vars,
        sumation_sets,
        grinding,
        true,
    )
}

fn verify_core<EF, F, Challenger>(
    verifier_state: &mut VerifierState<F, EF, Challenger>,
    max_degree_per_vars: &[usize],
    sumation_sets: Vec<Vec<EF>>,
    _grinding: SumcheckGrinding,
    first_round_tail_only: bool,
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
        // If this is the first round of a univariate-skip zerocheck, the prover
        // might have sent only the non-zero tail evaluations p(z) for z >= |sumation_set|,
        // with the prefix evaluations equal to zero. Detect and rebuild accordingly.
        let expect_eval_tail_only = first_round && first_round_tail_only;
        let pol = if expect_eval_tail_only {
            // Reconstruct lagrange interpolation from evaluations where the first
            // sumation_set.len() points are zeros, and the remaining are provided.
            // We read exactly (deg + 1 - sumation_set.len()) values: p(z) for z in [len..deg].
            let zero_prefix = sumation_set.len();
            let provided = verifier_state.next_extension_scalars_vec(deg + 1 - zero_prefix)?;
            let evals = (0..zero_prefix)
                .map(|i| (EF::from_usize(i), EF::ZERO))
                .chain(
                    (zero_prefix..=deg)
                        .zip(provided.into_iter())
                        .map(|(z, v)| (EF::from_usize(z), v)),
                )
                .collect::<Vec<_>>();
            WhirDensePolynomial::lagrange_interpolation(&evals).unwrap()
        } else {
            // General case: the prover now sends evaluations instead of coefficients.
            // Read deg+1 evaluations at x=0..deg and interpolate.
            let provided = verifier_state.next_extension_scalars_vec(deg + 1)?;
            let evals = (0..=deg)
                .zip(provided.into_iter())
                .map(|(z, v)| (EF::from_usize(z), v))
                .collect::<Vec<_>>();
            WhirDensePolynomial::lagrange_interpolation(&evals).unwrap()
        };

        let computed_sum = sumation_set.iter().map(|&s| pol.evaluate(s)).sum();
        if first_round {
            first_round = false;
            sum = computed_sum;
        } else if target != computed_sum {
            return Err(SumcheckError::InvalidRound);
        }
        let challenge = verifier_state.sample();

        // Grinding deactivated for optimization check
        // let pow_bits = grinding.pow_bits::<EF>(deg);
        // verifier_state.check_pow_grinding(pow_bits)?;

        target = pol.evaluate(challenge);
        challenges.push(challenge);
    }
    Ok((
        sum,
        Evaluation {
            point: challenges,
            value: target,
        },
    ))
}
