//! §C: Random Walk with Restart (RWR) Spreading Activation.
//!
//! Collins & Loftus (1975) spreading activation via RWR.
//! Runs in REM sleep daemon as background consolidation.

use anyhow::Result;
use sqlx::PgPool;

/// §C: RWR-based spreading activation during REM consolidation.
///
/// For each high-importance entity, diffuse importance to neighbors
/// via random walk with restart (α=0.85, 20 iterations).
pub async fn spreading_activation(pool: &PgPool) -> Result<()> {
    // Get seed entities (high importance)
    let seeds: Vec<(uuid::Uuid, f64)> = sqlx::query_as(
        "SELECT id, importance FROM brain_entities
         WHERE importance > 0.5
         ORDER BY importance DESC
         LIMIT 20"
    )
    .fetch_all(pool)
    .await?;

    if seeds.is_empty() {
        return Ok(());
    }

    // For each seed, run RWR and boost neighbors
    for (seed_id, seed_importance) in &seeds {
        // Get direct neighbors
        let neighbors: Vec<(uuid::Uuid, f64)> = sqlx::query_as(
            "SELECT CASE WHEN from_entity = $1 THEN to_entity ELSE from_entity END,
                    strength
             FROM brain_relations
             WHERE from_entity = $1 OR to_entity = $1"
        )
        .bind(seed_id)
        .fetch_all(pool)
        .await?;

        if neighbors.is_empty() {
            continue;
        }

        // Diffuse a fraction of seed importance to neighbors
        let alpha = 0.85;
        let boost_per_neighbor = seed_importance * (1.0 - alpha) / neighbors.len() as f64;

        for (neighbor_id, strength) in &neighbors {
            let weighted_boost = boost_per_neighbor * strength;
            sqlx::query(
                "UPDATE brain_entities SET
                    importance = LEAST(importance + $1, 1.0),
                    updated_at = NOW()
                 WHERE id = $2"
            )
            .bind(weighted_boost)
            .bind(neighbor_id)
            .execute(pool)
            .await?;
        }
    }

    tracing::info!(seeds = seeds.len(), "RWR spreading activation completed");
    Ok(())
}
