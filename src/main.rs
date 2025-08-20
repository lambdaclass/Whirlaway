#![cfg_attr(not(test), allow(unused_crate_dependencies))]

mod examples;

use air::AirSettings;
use whir_p3::parameters::{FoldingFactor, errors::SecurityAssumption};

use crate::examples::fibonacci::prove_fibonacci;
use crate::examples::poseidon2::prove_poseidon2;

const SECURITY_BITS: usize = 128;

fn main() {
    if std::env::var("EXAMPLE").ok().as_deref() == Some("fib") {
        let (log_n_rows, log_inv_rate) = (16, 1);
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
