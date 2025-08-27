use air::AirSettings;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tracing::Subscriber;
use tracing_subscriber::{Layer, Registry, layer::Context, prelude::*, registry::LookupSpan};

use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};
use whirlaway::examples::poseidon2::prove_poseidon2;

const LOG_N_ROWS: usize = 18;
const LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 128;
const UNIVARIATE_SKIPS: usize = 4;

#[derive(Copy, Clone, Eq, PartialEq)]
enum PhaseTag {
    None,
    Zerocheck,
    Inner,
}

struct SpanData {
    start: Option<Instant>,
    phase: PhaseTag,
}

struct Accumulators {
    zerocheck_ns: AtomicU64,
    inner_ns: AtomicU64,
}

impl Accumulators {
    fn new() -> Self {
        Self {
            zerocheck_ns: AtomicU64::new(0),
            inner_ns: AtomicU64::new(0),
        }
    }

    fn take_zerocheck(&self) -> Duration {
        let ns = self.zerocheck_ns.swap(0, Ordering::Relaxed);
        Duration::from_nanos(ns)
    }

    fn take_inner(&self) -> Duration {
        let ns = self.inner_ns.swap(0, Ordering::Relaxed);
        Duration::from_nanos(ns)
    }
}

thread_local! {
    static CURRENT_PHASE: std::cell::Cell<PhaseTag> = std::cell::Cell::new(PhaseTag::None);
}

struct SumcheckTimingLayer {
    acc: std::sync::Arc<Accumulators>,
}

impl SumcheckTimingLayer {
    fn new() -> (Self, std::sync::Arc<Accumulators>) {
        let acc = std::sync::Arc::new(Accumulators::new());
        (Self { acc: acc.clone() }, acc)
    }
}

impl<S> Layer<S> for SumcheckTimingLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::span::Id,
        ctx: Context<'_, S>,
    ) {
        if let Some(span) = ctx.span(id) {
            let meta = attrs.metadata();
            if meta.name() == "sumcheck_round" {
                let mut ext = span.extensions_mut();
                ext.insert(SpanData {
                    start: None,
                    phase: PhaseTag::None,
                });
            }
        }
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let meta = span.metadata();
            match meta.name() {
                "zerocheck" => CURRENT_PHASE.with(|p| p.set(PhaseTag::Zerocheck)),
                "inner sumchecks" => CURRENT_PHASE.with(|p| p.set(PhaseTag::Inner)),
                "sumcheck_round" => {
                    let mut ext = span.extensions_mut();
                    if let Some(data) = ext.get_mut::<SpanData>() {
                        data.phase = CURRENT_PHASE.with(|p| p.get());
                        data.start = Some(Instant::now());
                    }
                }
                _ => {}
            }
        }
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let meta = span.metadata();
            match meta.name() {
                "zerocheck" | "inner sumchecks" => CURRENT_PHASE.with(|p| p.set(PhaseTag::None)),
                "sumcheck_round" => {
                    let mut ext = span.extensions_mut();
                    if let Some(data) = ext.get_mut::<SpanData>() {
                        if let Some(start) = data.start.take() {
                            let dur = start.elapsed().as_nanos() as u64;
                            match data.phase {
                                PhaseTag::Zerocheck => {
                                    self.acc.zerocheck_ns.fetch_add(dur, Ordering::Relaxed);
                                }
                                PhaseTag::Inner => {
                                    self.acc.inner_ns.fetch_add(dur, Ordering::Relaxed);
                                }
                                PhaseTag::None => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

pub fn sumcheck_timings_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("sumcheck_timings");
    group
        .sample_size(20)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(20));

    let settings = AirSettings::new(
        SECURITY_BITS,
        SecurityAssumption::CapacityBound,
        FoldingFactor::ConstantFromSecondRound(7, 4),
        LOG_INV_RATE,
        UNIVARIATE_SKIPS,
        5,
    );

    group.bench_function(BenchmarkId::new("zerocheck_only", LOG_N_ROWS), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (layer, acc) = SumcheckTimingLayer::new();
                tracing::subscriber::with_default(Registry::default().with(layer), || {
                    let _ = prove_poseidon2(LOG_N_ROWS, settings.clone(), 0, false);
                });
                total += acc.take_zerocheck();
            }
            black_box(total)
        });
    });

    group.bench_function(BenchmarkId::new("inner_only", LOG_N_ROWS), |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (layer, acc) = SumcheckTimingLayer::new();
                tracing::subscriber::with_default(Registry::default().with(layer), || {
                    let _ = prove_poseidon2(LOG_N_ROWS, settings.clone(), 0, false);
                });
                total += acc.take_inner();
            }
            black_box(total)
        });
    });

    group.finish();
}

criterion_group!(benches, sumcheck_timings_bench);
criterion_main!(benches);
