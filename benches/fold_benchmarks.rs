use criterion::{Criterion, black_box, criterion_group, criterion_main};
use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_koala_bear::KoalaBear;
use utils::{
    batch_fold_multilinear_in_large_field, batch_fold_multilinear_in_large_field_no_skip,
    batch_fold_multilinear_in_small_field, batch_fold_multilinear_in_small_field_no_skip,
    fold_multilinear_in_large_field, fold_multilinear_in_large_field_new,
    fold_multilinear_in_large_field_no_skip, fold_multilinear_in_small_field,
    fold_multilinear_in_small_field_no_skip, fold_multilinear_packed, fold_multilinear_packed_new,
    univariate_selectors,
};
use whir_p3::poly::evals::EvaluationsList;

type F = KoalaBear;
type EF = BinomialExtensionField<F, 8>;

// This benchmark implements the following comparisons:
// 1) skip = 1, small-field
//    Compare: fold_multilinear_in_small_field_no_skip VS fold_multilinear_in_small_field
// 2) skip = 1, large-field
//    Compare: fold_multilinear_in_large_field_no_skip VS fold_multilinear_in_large_field
// 3) skip > 1 (we use 4), packed
//    Compare: fold_multilinear_packed_new VS fold_multilinear_packed
// 4) skip > 1 (we use 4), large-field
//    Compare: fold_multilinear_in_large_field_new VS fold_multilinear_in_large_field
// Additionally, optional batch comparisons are implemented,

fn generate_f_scalars(skips: usize, challenge: F) -> Vec<F> {
    let selectors = univariate_selectors::<F>(skips);
    selectors.iter().map(|s| s.evaluate(challenge)).collect()
}

fn generate_ef_scalars(skips: usize, challenge: EF) -> Vec<EF> {
    let selectors = univariate_selectors::<F>(skips);
    selectors.iter().map(|s| s.evaluate(challenge)).collect()
}

fn first_round_skip_4_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("first_round_skip_4");
    group.sample_size(20);

    let size = 1 << 16; // 2^16 evaluations
    let m = EvaluationsList::new((0..size).map(|i| F::from_usize(i)).collect::<Vec<_>>());

    // 4) large-field: _new VS standard
    group.bench_function("fold_multilinear_in_large_field", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = EF::from_usize(42 + z);
                let scalars = generate_ef_scalars(4, challenge);
                black_box(fold_multilinear_in_large_field(&m, &scalars));
            }
        });
    });

    group.bench_function("fold_multilinear_in_large_field_new", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = EF::from_usize(42 + z);
                let scalars = generate_ef_scalars(4, challenge);
                black_box(fold_multilinear_in_large_field_new(&m, &scalars));
            }
        });
    });

    // 3) packed: _new VS standard
    group.bench_function("fold_multilinear_packed", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = F::from_usize(42 + z);
                let scalars = generate_f_scalars(4, challenge);
                black_box(fold_multilinear_packed(&m, &scalars));
            }
        });
    });

    group.bench_function("fold_multilinear_packed_new", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = F::from_usize(42 + z);
                let scalars = generate_f_scalars(4, challenge);
                black_box(fold_multilinear_packed_new(&m, &scalars));
            }
        });
    });

    group.finish();
}

fn subsequent_rounds_skip_1_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("subsequent_rounds_skip_1");
    group.sample_size(20);

    let size = 1 << 16; // 2^16 evaluations
    let m = EvaluationsList::new((0..size).map(|i| F::from_usize(i)).collect::<Vec<_>>());

    // 2) large-field: no_skip VS standard
    group.bench_function("fold_multilinear_in_large_field", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = EF::from_usize(42 + z);
                let scalars = generate_ef_scalars(1, challenge); // [1 - s, s]
                black_box(fold_multilinear_in_large_field(&m, &scalars));
            }
        });
    });

    group.bench_function("fold_multilinear_in_large_field_no_skip", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = EF::from_usize(42 + z);
                let scalars = generate_ef_scalars(1, challenge); // [1 - s, s]
                black_box(fold_multilinear_in_large_field_no_skip(&m, &scalars));
            }
        });
    });

    // 1) small-field: no_skip VS standard
    group.bench_function("fold_multilinear_in_small_field", |b| {
        b.iter(|| {
            let m_ef =
                EvaluationsList::new((0..size).map(|i| EF::from_usize(i)).collect::<Vec<_>>());
            for z in 0..3 {
                let challenge = F::from_usize(42 + z);
                let scalars = generate_f_scalars(1, challenge); // [1 - z, z]
                black_box(fold_multilinear_in_small_field(&m_ef, &scalars));
            }
        });
    });

    group.bench_function("fold_multilinear_in_small_field_no_skip", |b| {
        b.iter(|| {
            let m_ef =
                EvaluationsList::new((0..size).map(|i| EF::from_usize(i)).collect::<Vec<_>>());
            for z in 0..3 {
                let challenge = F::from_usize(42 + z);
                let scalars = generate_f_scalars(1, challenge); // [1 - z, z]
                black_box(fold_multilinear_in_small_field_no_skip(&m_ef, &scalars));
            }
        });
    });

    group.finish();
}

fn batch_fold_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_fold");
    group.sample_size(20);

    let size = 1 << 16; // 2^16 evaluations
    let num_polys = 4;

    let polys: Vec<EvaluationsList<F>> = (0..num_polys)
        .map(|poly_idx| {
            EvaluationsList::new(
                (0..size)
                    .map(|i| F::from_usize(i + poly_idx * 1000))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let polys_refs: Vec<&EvaluationsList<F>> = polys.iter().collect();

    group.bench_function("batch_fold_multilinear_in_large_field_skip_4", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = EF::from_usize(42 + z);
                let scalars = generate_ef_scalars(4, challenge);
                black_box(batch_fold_multilinear_in_large_field(&polys_refs, &scalars));
            }
        });
    });
    group.bench_function(
        "batch_fold_multilinear_in_large_field_skip_4_no_skip",
        |b| {
            b.iter(|| {
                for z in 0..3 {
                    let challenge = EF::from_usize(42 + z);
                    let scalars = generate_ef_scalars(4, challenge);
                    black_box(batch_fold_multilinear_in_large_field_no_skip(
                        &polys_refs,
                        &scalars,
                    ));
                }
            });
        },
    );

    group.bench_function("batch_fold_multilinear_in_large_field_skip_1", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = EF::from_usize(42 + z);
                let scalars = generate_ef_scalars(1, challenge);
                black_box(batch_fold_multilinear_in_large_field(&polys_refs, &scalars));
            }
        });
    });
    group.bench_function(
        "batch_fold_multilinear_in_large_field_skip_1_no_skip",
        |b| {
            b.iter(|| {
                for z in 0..3 {
                    let challenge = EF::from_usize(42 + z);
                    let scalars = generate_ef_scalars(1, challenge);
                    black_box(batch_fold_multilinear_in_large_field_no_skip(
                        &polys_refs,
                        &scalars,
                    ));
                }
            });
        },
    );

    let polys_ef: Vec<EvaluationsList<EF>> = (0..num_polys)
        .map(|poly_idx| {
            EvaluationsList::new(
                (0..size)
                    .map(|i| EF::from_usize(i + poly_idx * 1000))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let polys_ef_refs: Vec<&EvaluationsList<EF>> = polys_ef.iter().collect();

    group.bench_function("batch_fold_multilinear_in_small_field_skip_1", |b| {
        b.iter(|| {
            for z in 0..3 {
                let challenge = F::from_usize(42 + z);
                let scalars = generate_f_scalars(1, challenge);
                black_box(batch_fold_multilinear_in_small_field(
                    &polys_ef_refs,
                    &scalars,
                ));
            }
        });
    });
    group.bench_function(
        "batch_fold_multilinear_in_small_field_skip_1_no_skip",
        |b| {
            b.iter(|| {
                for z in 0..3 {
                    let challenge = F::from_usize(42 + z);
                    let scalars = generate_f_scalars(1, challenge);
                    black_box(batch_fold_multilinear_in_small_field_no_skip(
                        &polys_ef_refs,
                        &scalars,
                    ));
                }
            });
        },
    );

    group.finish();
}

criterion_group!(
    benches,
    first_round_skip_4_benchmark,
    subsequent_rounds_skip_1_benchmark,
    batch_fold_benchmark,
);
criterion_main!(benches);
