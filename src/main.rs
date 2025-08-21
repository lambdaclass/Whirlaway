#![cfg_attr(not(test), allow(unused_crate_dependencies))]

mod examples;

use crate::examples::fibonacci::prove_fibonacci;
use crate::examples::poseidon2::prove_poseidon2;
use air::AirSettings;
use p3_field::PrimeCharacteristicRing;
use p3_koala_bear::KoalaBear as F;
use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};

const SECURITY_BITS: usize = 128;

fn main() {
    if std::env::var("EXAMPLE").ok().as_deref() == Some("fib") {
        let (log_n_rows, log_inv_rate) = (16, 1);
        // Compute expected F_N for the chosen N (N = 1 << log_n_rows)
        let n_rows = 1usize << log_n_rows;
        let mut a = F::ZERO; // F_0
        let mut b = F::ONE; // F_1
        for _ in 0..n_rows {
            let next = a + b;
            a = b;
            b = next;
        }
        let expected_last = a; // F_N
        let benchmark = prove_fibonacci(
            log_n_rows,
            AirSettings::new(
                SECURITY_BITS,
                SecurityAssumption::CapacityBound,
                FoldingFactor::ConstantFromSecondRound(7, 4),
                log_inv_rate,
                4,
                2,
            ),
            0,
            true,
            expected_last,
        );
        println!("\n{benchmark}");
        return;
    }

    let (log_n_rows, log_inv_rate) = (18, 1);
    let benchmark = prove_poseidon2(
        log_n_rows,
        AirSettings::new(
            SECURITY_BITS,
            SecurityAssumption::CapacityBound,
            FoldingFactor::ConstantFromSecondRound(7, 4),
            log_inv_rate,
            4,
            5,
        ),
        0,
        true,
    );
    println!("\n{benchmark}");
}
