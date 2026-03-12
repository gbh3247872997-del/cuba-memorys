//! Custom FSRS-6 implementation (ADR-001: ~50 LOC, zero deps).
//!
//! Implements retrievability calculation and stability update.
//! Based on Ye 2024 paper, 21 parameters.
//!
//! V4: FSRS-6 upgrade from FSRS-4 (21 params instead of 4).

use crate::constants::{DECAY_THRESHOLD, FSRS6_DEFAULT_PARAMS};
use anyhow::Result;
use sqlx::PgPool;

/// FSRS-6 decay constant.
const DECAY: f64 = -0.5;
/// FSRS-6 factor derived from decay constant.
const FACTOR: f64 = 0.9f64; // 19.0 / 81.0 precomputed below

/// Calculate retrievability from stability and elapsed days.
///
/// Formula: R(t, S) = (1 + FACTOR * t / S)^DECAY
///
/// Args:
///     stability: Current stability parameter (S).
///     elapsed_days: Days since last review/access.
///
/// Returns:
///     Retrievability in [0.0, 1.0].
pub fn retrievability(stability: f64, elapsed_days: f64) -> f64 {
    if stability <= 0.0 || elapsed_days < 0.0 {
        return 0.0;
    }
    let factor = 19.0_f64 / 81.0;
    (1.0 + factor * elapsed_days / stability).powf(DECAY)
}

/// Update stability after a recall event.
///
/// Uses FSRS-6 formula with 21 default parameters.
///
/// Args:
///     current_stability: Current S value.
///     difficulty: Current D value (1-10).
///     retrievability: Current R(t, S).
///     rating: How well recalled (0-3, where 3 = easy).
///
/// Returns:
///     New stability value.
pub fn update_stability(
    current_stability: f64,
    difficulty: f64,
    retrievability: f64,
    rating: u8,
) -> f64 {
    let w = &FSRS6_DEFAULT_PARAMS;

    // Clamp inputs
    let d = difficulty.clamp(1.0, 10.0);
    let r = retrievability.clamp(0.0, 1.0);
    let s = current_stability.max(0.01);

    match rating {
        // Again (forgot) → new stability from scratch
        0 => {
            w[11] * d.powf(-w[12]) * ((s + 1.0).powf(w[13]) - 1.0)
                * (w[14] * (1.0 - r)).exp()
        }
        // Hard/Good/Easy → multiply current stability
        _ => {
            let rating_bonus = match rating {
                1 => w[15], // Hard
                2 => 1.0,   // Good (baseline)
                3 => w[16], // Easy
                _ => 1.0,
            };
            s * (1.0 + (w[8] * d.powf(-w[9]) * (s.powf(-w[10]) - 1.0) * rating_bonus).exp())
        }
    }
}

/// Update difficulty based on rating.
///
/// Args:
///     current_difficulty: Current D value.
///     rating: How well recalled (0-3).
///
/// Returns:
///     New difficulty value in [1.0, 10.0].
pub fn update_difficulty(current_difficulty: f64, rating: u8) -> f64 {
    let w = &FSRS6_DEFAULT_PARAMS;
    let d = current_difficulty;

    // Mean reversion toward initial difficulty
    let delta = -(w[6] as f64) * (rating as f64 - 3.0);
    let new_d = d + delta;

    // Apply mean reversion
    let mean_reversion = w[7] * (w[4] - new_d);
    (new_d + mean_reversion).clamp(1.0, 10.0)
}

/// Batch decay — apply FSRS retrievability check on all observations.
///
/// V1: Single batch UPDATE instead of N individual queries.
///
/// Args:
///     pool: Database pool.
///     protected: Entity IDs to skip (active session protection).
///
/// Returns:
///     Number of observations with updated stability.
pub async fn batch_decay(pool: &PgPool, protected: &[uuid::Uuid]) -> Result<usize> {
    // Get observations eligible for decay (retrievability below threshold)
    let result = sqlx::query(
        r#"
        UPDATE brain_observations SET
            stability = stability * 0.95,
            retrieval_strength = retrieval_strength * 0.95,
            updated_at = NOW()
        WHERE entity_id NOT IN (SELECT UNNEST($1::uuid[]))
          AND observation_type != 'superseded'
          AND (
            -- Check if retrievability is below threshold
            -- R(t, S) = (1 + 19/81 * elapsed / stability)^(-0.5)
            POWER(1.0 + (19.0/81.0) * EXTRACT(EPOCH FROM (NOW() - last_accessed)) / 86400.0 / GREATEST(stability, 0.01), -0.5) < $2
          )
        "#,
    )
    .bind(protected)
    .bind(DECAY_THRESHOLD)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrievability_fresh() {
        // Just reviewed → R should be ~1.0
        let r = retrievability(1.0, 0.0);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_retrievability_decays() {
        // After 10 days with stability=1.0 → moderate (FSRS-6 uses slow decay)
        let r = retrievability(1.0, 10.0);
        assert!(r < 0.6, "R should decay: got {r}");
        assert!(r > 0.4, "R should not collapse: got {r}");
    }

    #[test]
    fn test_retrievability_high_stability() {
        // With stability=30, after 10 days → still high
        let r = retrievability(30.0, 10.0);
        assert!(r > 0.8);
    }

    #[test]
    fn test_update_stability_recall() {
        let new_s = update_stability(1.0, 5.0, 0.9, 2); // Good rating
        assert!(new_s > 1.0, "stability should increase on recall");
    }

    #[test]
    fn test_update_stability_forget() {
        let new_s = update_stability(10.0, 5.0, 0.3, 0); // Again rating
        assert!(new_s < 10.0, "stability should decrease on forget, got {new_s}");
        assert!(new_s > 0.0, "stability should remain positive");
    }

    #[test]
    fn test_difficulty_bounds() {
        let d = update_difficulty(5.0, 0); // Again → difficulty increases
        assert!(d >= 1.0 && d <= 10.0);
    }
}
