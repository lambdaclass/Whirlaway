use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use itertools::Itertools;
use whir_p3::fiat_shamir::domain_separator::DomainSeparator;

use air::table::AirTable;
use sumcheck::{SumcheckGrinding, sc_round};

use rand::{Rng, RngCore, SeedableRng, rng, rngs::StdRng};

use p3_challenger::DuplexChallenger;
use p3_field::BasedVectorSpace;
use p3_field::PrimeCharacteristicRing;
use p3_field::cyclic_subgroup_known_order;
use p3_field::extension::BinomialExtensionField;
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear, Poseidon2KoalaBear};
use p3_matrix::Matrix;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants, generate_trace_rows};
use whir_p3::poly::evals::EvaluationsList;

type Poseidon16 = Poseidon2KoalaBear<16>;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;
type LinearLayers = GenericPoseidon2LinearLayersKoalaBear;
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = 3;
const SBOX_REGISTERS: usize = 0;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

fn column_up(column: &EvaluationsList<F>) -> EvaluationsList<F> {
    let mut up = column.clone();
    up.evals_mut()[column.num_evals() - 1] = up.evals()[column.num_evals() - 2];
    up
}

fn column_down(column: &EvaluationsList<F>) -> EvaluationsList<F> {
    let mut down = column.evals()[1..].to_vec();
    down.push(*down.last().unwrap());
    EvaluationsList::new(down)
}

fn get_random_base_field_elements(log_n: usize) -> Vec<F> {
    let mut rng = rng();

    (0..(1 << log_n))
        .map(|_| {
            let val: u32 = rng.next_u32();
            F::new(val)
        })
        .collect()
}

fn get_random_extension_field_elements(n: usize) -> Vec<EF> {
    let mut rng = rng();

    (0..n)
        .map(|_| EF::from_basis_coefficients_fn(|_| F::new(rng.next_u32())))
        .collect()
}

fn bench_sc_round_first_with_skip(c: &mut Criterion) {
    let mut group = c.benchmark_group("sc_round");
    let mut rng = StdRng::seed_from_u64(0);

    let skip = 4;

    let log_n_rows = 24;
    let n_rows = 1 << log_n_rows;

    let constants =
        RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rng);

    let poseidon_air = Poseidon2Air::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >::new(constants.clone());

    let inputs: Vec<[F; WIDTH]> = (0..n_rows)
        .map(|_| std::array::from_fn(|_| rng.random()))
        .collect();

    let witness_matrix = generate_trace_rows::<
        F,
        LinearLayers,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >(inputs, &constants, 0)
    .transpose();

    let witness = witness_matrix
        .rows()
        .map(|col| whir_p3::poly::evals::EvaluationsList::new(col.collect()))
        .collect::<Vec<_>>();

    let table = AirTable::<F, EF, _>::new(poseidon_air, log_n_rows, skip, Vec::new(), 3);

    let multilinears: Vec<EvaluationsList<F>> = witness
        .iter()
        .map(|c| column_up(c))
        .chain(witness.iter().map(|c| column_down(c)))
        .collect();

    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);

    let mut n_vars: usize = 8;

    let eq_factor = get_random_extension_field_elements(log_n_rows + 1 - skip);

    let domainsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
    let challenger = MyChallenger::new(poseidon16);

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    let constraints_batching_scalar = prover_state.sample();

    let constraints_batching_scalars =
        cyclic_subgroup_known_order(constraints_batching_scalar, table.n_constraints)
            .collect::<Vec<_>>();

    let mut sum: EF = EF::ZERO;
    let mut challenges = Vec::new();
    let mut missing_mul_factor: Option<EF> = None;

    sc_round(
        skip,                               // skip
        &multilinears.iter().collect_vec(), // multilinears
        &mut n_vars,                        // n_vars
        &table.air,                         // computation
        Some(&eq_factor),                   // eq_factor
        &constraints_batching_scalars,      // batching_scalars
        true,                               // is_zerofier
        &mut prover_state,                  // fs_prover
        3,                                  // comp_degree
        &mut sum,                           // sum
        SumcheckGrinding::None,             // grinding
        &mut challenges,                    // challenges
        0,                                  // round
        &mut missing_mul_factor,            // missing_mul_factor
    );

    group.finish();
}

criterion_group!(benches, bench_sc_round_first_with_skip);

criterion_main!(benches);
