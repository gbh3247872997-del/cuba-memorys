//! V5: Prediction Error Gating — adaptive 3-threshold system.
//!
//! Determines how observations should be processed based on
//! prediction error (novelty vs expectation).
//!
//! V5.1: Adaptive thresholds based on EMA of recent similarities.
//! Inspired by Free Energy Principle (Friston, Nature 2023).

use crate::constants::{
    PRED_ERROR_REINFORCE,
    PRED_ERROR_UPDATE,
};

/// Action to take based on prediction error.
#[derive(Debug, Clone, PartialEq)]
pub enum GatingAction {
    /// High similarity (>reinforce threshold): reinforce existing (boost importance).
    Reinforce,
    /// Medium similarity (>update threshold): update existing observation.
    Update,
    /// Low similarity (<update threshold): create new observation (novel info).
    Create,
}

/// Determine action based on similarity score with static thresholds.
///
/// Three thresholds (V5):
/// - similarity > 0.92 → REINFORCE (just boost, don't duplicate)
/// - similarity > 0.75 → UPDATE (merge with existing)
/// - similarity ≤ 0.75 → CREATE (genuinely new information)
pub fn gate(similarity: f64) -> GatingAction {
    if similarity >= PRED_ERROR_REINFORCE {
        GatingAction::Reinforce
    } else if similarity >= PRED_ERROR_UPDATE {
        GatingAction::Update
    } else {
        GatingAction::Create
    }
}

/// V5.1: Adaptive gating based on distribution of recent similarities.
///
/// Instead of static thresholds, use EMA of similarity + std_dev to
/// dynamically adjust what constitutes "novel" vs "redundant".
///
/// Inspired by Free Energy Principle (Friston):
///   - System minimizes prediction error (surprise)
///   - Threshold adapts to current "expected" similarity level
///   - If most observations are highly similar → raise CREATE threshold
///   - If observations are diverse → lower CREATE threshold
///
/// Fallback: returns static thresholds if insufficient data.
pub fn adaptive_gate(similarity: f64, recent_similarities: &[f64]) -> GatingAction {
    let (reinforce_thresh, update_thresh) = adaptive_thresholds(recent_similarities);

    if similarity >= reinforce_thresh {
        GatingAction::Reinforce
    } else if similarity >= update_thresh {
        GatingAction::Update
    } else {
        GatingAction::Create
    }
}

/// Compute adaptive thresholds from recent similarity distribution.
///
/// Returns (reinforce_threshold, update_threshold).
///
/// Formula:
///   reinforce = min(0.98, mean + 1.5 * std_dev)
///   update    = min(reinforce - 0.05, mean + 0.5 * std_dev)
///
/// Minimum 5 samples required; fallback to static thresholds.
pub fn adaptive_thresholds(recent_similarities: &[f64]) -> (f64, f64) {
    if recent_similarities.len() < 5 {
        // Insufficient data → fallback to static
        return (PRED_ERROR_REINFORCE, PRED_ERROR_UPDATE);
    }

    let n = recent_similarities.len() as f64;
    let mean = recent_similarities.iter().sum::<f64>() / n;
    let variance = recent_similarities.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / n;
    let std_dev = variance.sqrt();

    // Adaptive thresholds clamped to sensible ranges
    let reinforce = (mean + 1.5 * std_dev).clamp(0.80, 0.98);
    let update = (mean + 0.5 * std_dev).clamp(0.50, reinforce - 0.05);

    (reinforce, update)
}

/// Check if a new observation is novel enough to store.
///
/// Returns (should_store, highest_similarity, action).
pub fn assess_novelty(similarity_scores: &[f64]) -> (bool, f64, GatingAction) {
    let max_sim = similarity_scores.iter().cloned().fold(0.0f64, f64::max);
    let action = gate(max_sim);

    let should_store = match action {
        GatingAction::Reinforce => false, // Already exists, just boost
        GatingAction::Update => true,     // Update existing
        GatingAction::Create => true,     // Novel, create new
    };

    (should_store, max_sim, action)
}

/// V5.1: Adaptive novelty assessment using recent similarity distribution.
///
/// Returns (should_store, highest_similarity, action).
pub fn assess_novelty_adaptive(
    similarity_scores: &[f64],
    recent_similarities: &[f64],
) -> (bool, f64, GatingAction) {
    let max_sim = similarity_scores.iter().cloned().fold(0.0f64, f64::max);
    let action = adaptive_gate(max_sim, recent_similarities);

    let should_store = match action {
        GatingAction::Reinforce => false,
        GatingAction::Update => true,
        GatingAction::Create => true,
    };

    (should_store, max_sim, action)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_reinforce() {
        assert_eq!(gate(0.95), GatingAction::Reinforce);
        assert_eq!(gate(0.92), GatingAction::Reinforce);
    }

    #[test]
    fn test_gate_update() {
        assert_eq!(gate(0.85), GatingAction::Update);
        assert_eq!(gate(0.75), GatingAction::Update);
    }

    #[test]
    fn test_gate_create() {
        assert_eq!(gate(0.74), GatingAction::Create);
        assert_eq!(gate(0.5), GatingAction::Create);
        assert_eq!(gate(0.0), GatingAction::Create);
    }

    #[test]
    fn test_novelty_assessment() {
        // High similarity → don't store
        let (store, max_sim, action) = assess_novelty(&[0.95, 0.80, 0.60]);
        assert!(!store);
        assert!((max_sim - 0.95).abs() < 0.001);
        assert_eq!(action, GatingAction::Reinforce);

        // Low similarity → store
        let (store, max_sim, action) = assess_novelty(&[0.3, 0.2, 0.1]);
        assert!(store);
        assert!((max_sim - 0.3).abs() < 0.001);
        assert_eq!(action, GatingAction::Create);

        // Empty → store (no existing data)
        let (store, _, action) = assess_novelty(&[]);
        assert!(store);
        assert_eq!(action, GatingAction::Create);
    }

    #[test]
    fn test_adaptive_thresholds_insufficient_data() {
        // < 5 samples → returns static
        let (r, u) = adaptive_thresholds(&[0.5, 0.6]);
        assert!((r - PRED_ERROR_REINFORCE).abs() < 0.001);
        assert!((u - PRED_ERROR_UPDATE).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_thresholds_high_similarity_dist() {
        // All recent observations are very similar → raise thresholds
        let recent = vec![0.90, 0.88, 0.92, 0.89, 0.91, 0.90, 0.88];
        let (r, u) = adaptive_thresholds(&recent);
        assert!(r > PRED_ERROR_REINFORCE - 0.05, "reinforce should be high: {r}");
        assert!(u > 0.75, "update should reflect high baseline: {u}");
    }

    #[test]
    fn test_adaptive_thresholds_diverse_dist() {
        // Very diverse observations → lower thresholds
        let recent = vec![0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4];
        let (r, u) = adaptive_thresholds(&recent);
        assert!(r < 0.95, "reinforce threshold should accommodate diversity: {r}");
        assert!(u < PRED_ERROR_UPDATE + 0.1, "update should be reasonable: {u}");
    }

    #[test]
    fn test_adaptive_gate_novel_in_uniform_dist() {
        // Distribution of high similarities → a low-similarity observation is truly novel
        let recent = vec![0.85, 0.88, 0.87, 0.86, 0.89, 0.90, 0.85];
        let action = adaptive_gate(0.5, &recent);
        assert_eq!(action, GatingAction::Create);
    }

    #[test]
    fn test_adaptive_gate_redundant_in_uniform_dist() {
        // Distribution of high similarities → a similar observation is redundant
        let recent = vec![0.85, 0.88, 0.87, 0.86, 0.89, 0.90, 0.85];
        let action = adaptive_gate(0.95, &recent);
        assert_eq!(action, GatingAction::Reinforce);
    }
}
