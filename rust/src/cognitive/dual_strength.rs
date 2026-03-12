//! Dual-Strength Model (Bjork & Bjork 1992) — ADR-002.
//!
//! Storage strength: only increases (encoding quality).
//! Retrieval strength: decays with time (accessibility).
//!
//! VF1: Dual-Strength Model
//! VF2: Testing Effect (search > creation for strengthening)

use crate::constants::{
    RETRIEVAL_DECAY_FACTOR, RETRIEVAL_SEARCH_BOOST, STORAGE_STRENGTH_INCREMENT,
};
use anyhow::Result;
use sqlx::PgPool;

/// Update storage strength on access (only increases).
///
/// Formula: SS_new = SS + 0.1 * (1.0 - SS)
/// This creates diminishing returns as SS approaches 1.0.
pub fn increment_storage(current: f64) -> f64 {
    let delta = STORAGE_STRENGTH_INCREMENT * (1.0 - current);
    (current + delta).min(1.0)
}

/// Decay retrieval strength based on elapsed time.
///
/// Formula: RS_new = RS * 0.95^days
pub fn decay_retrieval(current: f64, elapsed_days: f64) -> f64 {
    if elapsed_days <= 0.0 {
        return current;
    }
    (current * RETRIEVAL_DECAY_FACTOR.powf(elapsed_days)).max(0.0)
}

/// VF2: Testing Effect — search boosts retrieval more than creation.
///
/// Retrieval (actively finding a memory) strengthens it more
/// than passive study (creating/updating) — Roediger & Karpicke 2006.
pub fn search_boost_retrieval(current: f64) -> f64 {
    (current + RETRIEVAL_SEARCH_BOOST).min(1.0)
}

/// Memory state classification based on dual-strength values.
///
/// VF4 preparation: Active(≥70%), Dormant(40-70%), Silent(10-40%), Unavailable(<10%).
pub fn memory_state(storage: f64, retrieval: f64) -> &'static str {
    let composite = 0.5 * retrieval + 0.3 * storage + 0.2 * retrieval.powf(0.5);
    if composite >= 0.70 {
        "active"
    } else if composite >= 0.40 {
        "dormant"
    } else if composite >= 0.10 {
        "silent"
    } else {
        "unavailable"
    }
}

/// Batch update dual-strength on entity access.
pub async fn on_entity_access(pool: &PgPool, entity_id: uuid::Uuid) -> Result<()> {
    sqlx::query(
        r#"
        UPDATE brain_observations SET
            storage_strength = LEAST(storage_strength + 0.1 * (1.0 - storage_strength), 1.0),
            retrieval_strength = 1.0,
            last_accessed = NOW()
        WHERE entity_id = $1
          AND observation_type != 'superseded'
        "#,
    )
    .bind(entity_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// VF2: Boost retrieval on search match (Testing Effect).
pub async fn on_search_match(pool: &PgPool, observation_ids: &[uuid::Uuid]) -> Result<()> {
    if observation_ids.is_empty() {
        return Ok(());
    }
    sqlx::query(
        r#"
        UPDATE brain_observations SET
            retrieval_strength = LEAST(retrieval_strength + $1, 1.0),
            last_accessed = NOW()
        WHERE id = ANY($2)
        "#,
    )
    .bind(RETRIEVAL_SEARCH_BOOST)
    .bind(observation_ids)
    .execute(pool)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_only_increases() {
        let s = increment_storage(0.5);
        assert!(s > 0.5);
        assert!(s <= 1.0);
    }

    #[test]
    fn test_storage_diminishing_returns() {
        let s1 = increment_storage(0.1); // low → big jump
        let s2 = increment_storage(0.9); // high → small jump
        assert!(s1 - 0.1 > s2 - 0.9);
    }

    #[test]
    fn test_retrieval_decays() {
        let r = decay_retrieval(1.0, 10.0);
        assert!(r < 1.0);
        assert!(r > 0.0);
    }

    #[test]
    fn test_memory_states() {
        assert_eq!(memory_state(1.0, 1.0), "active");
        assert_eq!(memory_state(0.5, 0.3), "dormant");
        assert_eq!(memory_state(0.2, 0.15), "silent");
        assert_eq!(memory_state(0.0, 0.0), "unavailable");
    }
}
