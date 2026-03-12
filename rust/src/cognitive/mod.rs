//! Cognitive module — FSRS-6, Hebbian learning, Dual-Strength, density.
//!
//! These are library functions used by handlers and the REM daemon.
//! Many are not yet called from the handler layer but are tested and
//! ready for incremental wiring.
#![allow(dead_code)]

pub mod density;
pub mod dual_strength;
pub mod fsrs;
pub mod hebbian;
pub mod prediction_error;
pub mod spreading;
