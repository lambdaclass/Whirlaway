use criterion::{Criterion, black_box, criterion_group, criterion_main};
use utils::{
    batch_fold_multilinear_in_large_field, batch_fold_multilinear_in_small_field,
    univariate_selectors,
};
use whir_p3::poly::evals::EvaluationsList;

// Queremos hacer 4 comparaciones:

// 1) Comparar fold_multilinear_in_small_field_no_skip VS fold_multilinear_in_small_field para el caso
// skip = 1, es decir scalars = [1-z, z].

// 2) Comparar fold_multilinear_in_large_field_no_skip VS fold_multilinear_in_large_field para el caso
// skip = 1, es decir scalars = [1-s, s].

// 3) Comparar fold_multilk inear_packed_new VS fold_multilinear_packed. Esto se usa en el caso
// skip > 1.

// 4) Comparar fold_multilinear_in_large_field_new VS fold_multilinear_in_large_field. Esto se usa en el caso
// skip > 1.

fn packed_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("packed_fold_multilinear");

    group.sample_size(10);

    group.bench_function("fold_multilinear_in_small_field_no_skip", |b| {
        b.iter(|| {
            let m = EvaluationsList::new(vec![1.0, 2.0, 3.0, 4.0]);
            for z in 0..3 {
                let scalars = vec![F::ONE - F::from_usize(z), F::from_usize(z)];
                fold_multilinear_in_small_field_no_skip(&m, &scalars);
            }
        });
    });

    group.bench_function("fold_multilinear_in_small_field", |b| {
        b.iter(|| {
            let m = EvaluationsList::new(vec![1.0, 2.0, 3.0, 4.0]);
            for z in 0..3 {
                let scalars = vec![F::ONE - F::from_usize(z), F::from_usize(z)];
                fold_multilinear_in_small_field(&m, &scalars);
            }
        });
    });

    group.finish();
}
