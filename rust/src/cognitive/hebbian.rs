//! Hebbian learning — Oja's rule with BCM metaplastic throttling.
//!
//! BCM Theory (Bienenstock-Cooper-Munro, 1982): sliding threshold θ_M
//! adapts to recent activity, preventing runaway potentiation.
//!
//! When a node is accessed frequently, θ_M rises → boost decreases.
//! When a node is idle, θ_M decays → boost normalizes.

use anyhow::Result;
use sqlx::PgPool;

use crate::constants::{
    HEBBIAN_ACCESS_BOOST, HEBBIAN_SEARCH_BOOST, HEBBIAN_OJA_RATE,
    BCM_THROTTLE_SCALE, BCM_HIGH_ACTIVITY_THRESHOLD,
};

/// Boost entity importance on access with BCM throttling.
///
/// BCM sliding threshold: boost is scaled down when access_count is high
/// relative to the throttle threshold. This prevents "winner-take-all"
/// dynamics where popular nodes suppress the rest of the graph.
///
/// Formula: effective_boost = base_boost * throttle_factor
///   throttle_factor = max(0.1, 1.0 - (access_count / threshold) * scale)
pub async fn boost_on_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    // Read current access_count to compute BCM throttle
    let row: Option<(i32,)> = sqlx::query_as(
        "SELECT access_count FROM brain_entities WHERE id = $1"
    )
    .bind(entity_id)
    .fetch_optional(pool)
    .await?;

    let access_count = row.map(|(c,)| c).unwrap_or(0);
    let effective_boost = bcm_throttle(HEBBIAN_ACCESS_BOOST, access_count);

    sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(importance + $1, 1.0),
            access_count = access_count + 1,
            updated_at = NOW()
         WHERE id = $2"
    )
    .bind(effective_boost)
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Boost entity importance on search match (Testing Effect — VF2)
/// with BCM throttling.
pub async fn boost_on_search(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    let row: Option<(i32,)> = sqlx::query_as(
        "SELECT access_count FROM brain_entities WHERE id = $1"
    )
    .bind(entity_id)
    .fetch_optional(pool)
    .await?;

    let access_count = row.map(|(c,)| c).unwrap_or(0);
    let effective_boost = bcm_throttle(HEBBIAN_SEARCH_BOOST, access_count);

    sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(importance + $1, 1.0),
            access_count = access_count + 1,
            updated_at = NOW()
         WHERE id = $2"
    )
    .bind(effective_boost)
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Boost observation importance using Oja's rule.
///
/// Oja's rule: Δw = η * x * (y - w*x) ≈ η * boost for positive
/// Anti-Oja: w = max(w * decay, minimum) for negative
pub async fn oja_boost(pool: &PgPool, observation_id: uuid::Uuid, positive: bool) -> Result<()> {
    if positive {
        // Oja: increase bounded at 1.0
        sqlx::query(
            "UPDATE brain_observations SET
                importance = LEAST(importance + $1, 1.0),
                updated_at = NOW()
             WHERE id = $2"
        )
        .bind(HEBBIAN_OJA_RATE)
        .bind(observation_id)
        .execute(pool)
        .await?;
    } else {
        // Anti-Oja: decrease bounded at 0.0
        sqlx::query(
            "UPDATE brain_observations SET
                importance = GREATEST(importance * 0.8, 0.0),
                updated_at = NOW()
             WHERE id = $2"
        )
        .bind(observation_id)
        .execute(pool)
        .await?;
    }
    Ok(())
}

/// Strengthen relation on traversal (Hebbian synapse strengthening).
pub async fn strengthen_relation(pool: &PgPool, from_entity: uuid::Uuid, to_entity: uuid::Uuid) -> Result<()> {
    sqlx::query(
        "UPDATE brain_relations SET
            strength = LEAST(strength + $1, 1.0),
            updated_at = NOW()
         WHERE from_entity = $2 AND to_entity = $3"
    )
    .bind(HEBBIAN_OJA_RATE)
    .bind(from_entity)
    .bind(to_entity)
    .execute(pool)
    .await?;
    Ok(())
}

/// Boost neighbors via spreading activation (Collins & Loftus 1975).
pub async fn boost_neighbors(pool: &PgPool, entity_id: uuid::Uuid) -> Result<usize> {
    let result = sqlx::query(
        "UPDATE brain_entities SET
            importance = LEAST(importance + $1 * 0.5, 1.0),
            updated_at = NOW()
         WHERE id IN (
             SELECT CASE
                 WHEN from_entity = $2 THEN to_entity
                 ELSE from_entity
             END
             FROM brain_relations
             WHERE from_entity = $2 OR to_entity = $2
         )"
    )
    .bind(HEBBIAN_ACCESS_BOOST)
    .bind(entity_id)
    .execute(pool)
    .await?;

    Ok(result.rows_affected() as usize)
}

// ── BCM Utilities ────────────────────────────────────────────────

/// BCM sliding threshold — compute effective boost with throttling.
///
/// Bienenstock-Cooper-Munro (1982): high recent activity → lower learning rate.
/// This is a simplified version using access_count as the activity proxy.
///
/// Formula:
///   throttle = max(0.1, 1.0 - (access_count / threshold) * scale)
///   effective_boost = base_boost * throttle
fn bcm_throttle(base_boost: f64, access_count: i32) -> f64 {
    let activity_ratio = access_count as f64 / BCM_HIGH_ACTIVITY_THRESHOLD;
    let throttle = (1.0 - activity_ratio * BCM_THROTTLE_SCALE).max(0.1);
    base_boost * throttle
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_valid() {
        assert!(HEBBIAN_ACCESS_BOOST > 0.0 && HEBBIAN_ACCESS_BOOST < 1.0);
        assert!(HEBBIAN_SEARCH_BOOST > 0.0 && HEBBIAN_SEARCH_BOOST < 1.0);
        assert!(HEBBIAN_OJA_RATE > 0.0 && HEBBIAN_OJA_RATE < 1.0);
    }

    #[test]
    fn test_bcm_throttle_low_activity() {
        // Few accesses → near-full boost
        let boost = bcm_throttle(0.01, 5);
        assert!(boost > 0.008, "low activity should give near-full boost: {boost}");
    }

    #[test]
    fn test_bcm_throttle_high_activity() {
        // Many accesses → throttled boost
        let boost = bcm_throttle(0.01, 100);
        assert!(boost < 0.005, "high activity should throttle: {boost}");
        assert!(boost > 0.0, "boost should never be negative");
    }

    #[test]
    fn test_bcm_throttle_extreme() {
        // Extreme accesses → minimum floor (0.1 * base)
        let boost = bcm_throttle(0.01, 10_000);
        assert!((boost - 0.001).abs() < 0.0001, "extreme activity → 10% floor: {boost}");
    }
}
