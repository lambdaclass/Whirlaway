use ::air::AirSettings;
use air::table::AirTable;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_challenger::DuplexChallenger;
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_koala_bear::{KoalaBear, Poseidon2KoalaBear};
use p3_matrix::Matrix;
use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
use rand::{SeedableRng, rngs::StdRng};
use std::fmt;
use std::time::{Duration, Instant};
use tracing::level_filters::LevelFilter;
use tracing_forest::ForestLayer;
use tracing_subscriber::{EnvFilter, Registry, layer::SubscriberExt, util::SubscriberInitExt};
use whir_p3::{fiat_shamir::domain_separator::DomainSeparator, whir::parameters::WhirConfig};

// Field / permutation choices (reuse Poseidon2 sponge for Merkle/Challenger)
type Poseidon16 = Poseidon2KoalaBear<16>;
type Poseidon24 = Poseidon2KoalaBear<24>;

type MerkleHash = PaddingFreeSponge<Poseidon24, 24, 16, 8>; // leaf hashing
type MerkleCompress = TruncatedPermutation<Poseidon16, 2, 8, 16>; // 2-to-1 compression
type MyChallenger = DuplexChallenger<F, Poseidon16, 16, 8>;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;

#[derive(Clone, Debug)]
pub struct FibonacciBenchmark {
    pub log_n_rows: usize,
    pub settings: AirSettings,
    pub prover_time: Duration,
    pub verifier_time: Duration,
    pub proof_size: f64, // in bytes
}

impl fmt::Display for FibonacciBenchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Fibonacci - Security level: {} bits ({:?}), starting rate: 1/{}",
            self.settings.security_bits,
            self.settings.whir_soudness_type,
            1 << self.settings.whir_log_inv_rate,
        )?;
        let n_rows = 1 << self.log_n_rows;
        writeln!(
            f,
            "Proved {} Fibonacci steps in {:.3} s ({} / s)",
            n_rows,
            self.prover_time.as_millis() as f64 / 1000.0,
            (n_rows as f64 / self.prover_time.as_secs_f64()).round() as usize
        )?;
        writeln!(f, "Proof size: {:.1} KiB", self.proof_size / 1024.0)?;
        writeln!(f, "Verification: {} ms", self.verifier_time.as_millis())
    }
}

#[derive(Clone, Debug, Default)]
pub struct FibonacciAir;

impl<Fld: p3_field::Field> BaseAir<Fld> for FibonacciAir {
    fn width(&self) -> usize {
        // 6 columns total: [preprocessed transition_selector, preprocessed first_row_selector, preprocessed last_row_selector, preprocessed pub_x, F_r, F_{r+1}]
        6
    }
}

impl<B> Air<B> for FibonacciAir
where
    B: AirBuilder,
    B::M: Matrix<B::Var>,
{
    fn eval(&self, builder: &mut B) {
        let m = builder.main();

        // Column order:
        // 0 = transition_selector, 1 = first_row_selector, 2 = last_row_selector, 3 = pub_x, 4 = F_r, 5 = F_{r+1}
        let sel_trans_up: B::Var = m.get(0, 0).expect("in-bounds");
        let sel_first_up: B::Var = m.get(0, 1).expect("in-bounds");
        let sel_last_up: B::Var = m.get(0, 2).expect("in-bounds");
        let pub_x_up: B::Var = m.get(0, 3).expect("in-bounds");
        let a_up: B::Var = m.get(0, 4).expect("in-bounds");
        let b_up: B::Var = m.get(0, 5).expect("in-bounds");
        let a_down: B::Var = m.get(1, 4).expect("in-bounds");
        let b_down: B::Var = m.get(1, 5).expect("in-bounds");

        // Transition constraints multiplied by transition selector
        builder.assert_zero((a_down.clone() - b_up.clone()) * sel_trans_up.clone());
        builder.assert_zero((b_down - (a_up.clone() + b_up.clone())) * sel_trans_up);

        // Boundary constraints at first row: a_0 = 0, b_0 = 1
        builder.assert_zero(a_up * sel_first_up.clone());
        builder.assert_zero((b_up.clone() - B::Expr::ONE) * sel_first_up.clone());

        // Public value at last row: b_last == pub_x
        builder.assert_zero((b_up - pub_x_up) * sel_last_up);
    }
}

pub fn prove_fibonacci(
    log_n_rows: usize,
    settings: AirSettings,
    _n_preprocessed_columns: usize,
    display_logs: bool,
) -> FibonacciBenchmark {
    if display_logs {
        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }

    let n_rows = 1 << log_n_rows;

    // Generate Fibonacci witness columns: c0[r] = F_r, c1[r] = F_{r+1}
    let mut c0 = Vec::with_capacity(n_rows);
    let mut c1 = Vec::with_capacity(n_rows);
    let mut a = F::ZERO; // F_0
    let mut b = F::ONE; // F_1
    for _ in 0..n_rows {
        c0.push(a);
        c1.push(b);
        let next = a + b;
        a = b;
        b = next;
    }

    // Preprocessed selectors & public value column
    // transition selector: 1 on rows [0..N-1), 0 at last row
    let mut sel_trans = vec![F::ONE; n_rows];
    if let Some(last) = sel_trans.last_mut() {
        *last = F::ZERO;
    }
    // first row selector: 1 only at row 0
    let mut sel_first = vec![F::ZERO; n_rows];
    if !sel_first.is_empty() {
        sel_first[0] = F::ONE;
    }

    // last row selector: 1 only at last row
    let mut sel_last = vec![F::ZERO; n_rows];
    if let Some(last) = sel_last.last_mut() {
        *last = F::ONE;
    }
    // pub_x column: expected right value at last row, replicated for convenience
    let pub_x_val = *c1.last().unwrap_or(&F::ZERO);
    let pub_x = vec![pub_x_val; n_rows];

    let preprocessed_columns = vec![
        whir_p3::poly::evals::EvaluationsList::new(sel_trans),
        whir_p3::poly::evals::EvaluationsList::new(sel_first),
        whir_p3::poly::evals::EvaluationsList::new(sel_last),
        whir_p3::poly::evals::EvaluationsList::new(pub_x),
    ];

    let witness = vec![
        whir_p3::poly::evals::EvaluationsList::new(c0),
        whir_p3::poly::evals::EvaluationsList::new(c1),
    ];

    let air = FibonacciAir::default();
    let table = AirTable::<F, EF, _>::new(
        air,
        log_n_rows,
        settings.univariate_skips,
        preprocessed_columns,
        2, // casamos mejor con el shape del sumcheck cuando hay skip>=1
    );

    // Merkle and challenger setup
    let mut rng = StdRng::seed_from_u64(0);
    let poseidon16 = Poseidon16::new_from_rng_128(&mut rng);
    let poseidon24 = Poseidon24::new_from_rng_128(&mut rng);
    let merkle_hash = MerkleHash::new(poseidon24);
    let merkle_compress = MerkleCompress::new(poseidon16.clone());

    let t = Instant::now();

    let whir_params: WhirConfig<_, _, _, _, MyChallenger> =
        table.build_whir_params(&settings, merkle_hash.clone(), merkle_compress.clone());
    let mut domainsep: DomainSeparator<EF, F> = DomainSeparator::new(vec![]);
    domainsep.commit_statement::<_, _, _, 8>(&whir_params);
    domainsep.add_whir_proof::<_, _, _, 8>(&whir_params);

    let challenger = MyChallenger::new(poseidon16);

    let mut prover_state = domainsep.to_prover_state(challenger.clone());

    table.prove(
        &settings,
        merkle_hash.clone(),
        merkle_compress.clone(),
        &mut prover_state,
        witness,
    );

    let prover_time = t.elapsed();
    let time = Instant::now();

    let mut verifier_state =
        domainsep.to_verifier_state(prover_state.proof_data().to_vec(), challenger);

    table
        .verify(
            &settings,
            merkle_hash,
            merkle_compress,
            &mut verifier_state,
            log_n_rows,
        )
        .unwrap();
    let verifier_time = time.elapsed();

    let proof_size = prover_state.proof_data().len() as f64 * (F::ORDER_U64 as f64).log2() / 8.0;

    FibonacciBenchmark {
        log_n_rows,
        settings,
        prover_time,
        verifier_time,
        proof_size,
    }
}
