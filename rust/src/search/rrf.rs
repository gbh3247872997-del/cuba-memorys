//! §A: Weighted RRF Entropy Routing.
//!
//! Reciprocal Rank Fusion with Shannon entropy-based dynamic weighting.
//! V2: Post-fusion dedup removes semantic duplicates across signals.
//! V3: Adaptive k parameter (Azure AI Search 2025).

use std::collections::{HashMap, HashSet};

use crate::constants::{RRF_K, RRF_K_MIN, RRF_K_MAX};

/// A ranked search result.
#[derive(Clone, Debug)]
pub struct RankedResult {
    pub id: String,
    pub content: String,
    pub score: f64,
    pub source: String, // Which signal produced this
}

/// §A: Compute Shannon entropy of query for dynamic weight routing.
pub fn query_entropy(query: &str) -> f64 {
    let words: Vec<&str> = query.split_whitespace().collect();
    let total = words.len();
    if total == 0 {
        return 0.0;
    }
    let unique: HashSet<&str> = words.iter().copied().collect();
    let mut entropy = 0.0;
    for word in &unique {
        let freq = words.iter().filter(|w| *w == word).count() as f64 / total as f64;
        if freq > 0.0 {
            entropy -= freq * freq.log2();
        }
    }
    entropy
}

/// RRF fusion across N ranked signal lists.
///
/// Each signal can have a custom weight (§A: entropy-based).
/// V3: Uses adaptive k parameter — pass `None` for default (60.0).
pub fn fuse(
    signals: &[(Vec<RankedResult>, f64)], // (results, weight)
    dedup_threshold: f64,
    adaptive_k: Option<f64>,
) -> Vec<RankedResult> {
    let k = adaptive_k.unwrap_or(RRF_K);
    let mut scores: HashMap<String, f64> = HashMap::new();
    let mut items: HashMap<String, RankedResult> = HashMap::new();

    for (results, weight) in signals {
        for (rank, result) in results.iter().enumerate() {
            let rrf_score = weight / (k + rank as f64 + 1.0);
            *scores.entry(result.id.clone()).or_default() += rrf_score;
            items.entry(result.id.clone()).or_insert_with(|| result.clone());
        }
    }

    // Sort by fused score
    let mut sorted: Vec<(String, f64)> = scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // V2: Post-fusion dedup by content overlap
    let mut unique: Vec<RankedResult> = Vec::new();
    for (id, score) in sorted {
        if let Some(mut item) = items.remove(&id) {
            let is_dup = unique.iter().any(|existing| {
                text_overlap(&item.content, &existing.content) > dedup_threshold
            });
            if !is_dup {
                item.score = score;
                unique.push(item);
            }
        }
    }

    unique
}

/// V3: Compute adaptive RRF k from result count (Azure AI Search 2025).
///
/// With fewer results, lower k sharpens ranking discrimination.
pub fn adaptive_k(result_count: usize) -> f64 {
    (result_count as f64 * 0.5).clamp(RRF_K_MIN, RRF_K_MAX)
}

/// V2: Word-overlap ratio (Jaccard-like with min denominator).
fn text_overlap(a: &str, b: &str) -> f64 {
    let lower_a = a.to_lowercase();
    let lower_b = b.to_lowercase();
    let words_a: HashSet<&str> = lower_a.split_whitespace().collect();
    let words_b: HashSet<&str> = lower_b.split_whitespace().collect();
    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }
    let intersection = words_a.intersection(&words_b).count();
    intersection as f64 / words_a.len().min(words_b.len()) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_entropy_uniform() {
        // All unique words → high entropy
        let e = query_entropy("rust is fast and safe for systems programming");
        assert!(e > 2.5, "diverse query should have high entropy: got {e}");
    }

    #[test]
    fn test_query_entropy_repetitive() {
        let e = query_entropy("hello hello hello");
        assert!(e < 0.01, "repetitive query should have near-zero entropy: got {e}");
    }

    #[test]
    fn test_text_overlap_identical() {
        assert!((text_overlap("hello world", "hello world") - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_text_overlap_disjoint() {
        assert_eq!(text_overlap("hello world", "foo bar"), 0.0);
    }

    #[test]
    fn test_rrf_fusion_basic() {
        let signal1 = vec![
            RankedResult { id: "a".into(), content: "alpha".into(), score: 0.0, source: "text".into() },
            RankedResult { id: "b".into(), content: "beta".into(), score: 0.0, source: "text".into() },
        ];
        let signal2 = vec![
            RankedResult { id: "b".into(), content: "beta".into(), score: 0.0, source: "vec".into() },
            RankedResult { id: "c".into(), content: "gamma".into(), score: 0.0, source: "vec".into() },
        ];

        let fused = fuse(&[(signal1, 0.5), (signal2, 0.5)], 0.75, None);
        assert!(!fused.is_empty());
        // "b" appears in both signals, should rank first
        assert_eq!(fused[0].id, "b", "item in both signals should rank first");
    }

    #[test]
    fn test_adaptive_k_few_results() {
        // V3: With few results, k should be low (sharper ranking)
        let k = adaptive_k(10);
        assert_eq!(k, 10.0); // 10 * 0.5 = 5.0, clamped to min 10.0
    }

    #[test]
    fn test_adaptive_k_many_results() {
        // V3: With many results, k should be capped at 60
        let k = adaptive_k(200);
        assert_eq!(k, 60.0); // 200 * 0.5 = 100.0, clamped to max 60.0
    }

    #[test]
    fn test_adaptive_k_medium_results() {
        // V3: Mid-range results scale linearly
        let k = adaptive_k(60);
        assert_eq!(k, 30.0); // 60 * 0.5 = 30.0
    }

    #[test]
    fn test_rrf_fusion_with_adaptive_k() {
        let signal1 = vec![
            RankedResult { id: "a".into(), content: "alpha".into(), score: 0.0, source: "text".into() },
        ];
        let signal2 = vec![
            RankedResult { id: "a".into(), content: "alpha".into(), score: 0.0, source: "vec".into() },
        ];

        // Low k should give higher scores than high k
        let fused_low_k = fuse(&[(signal1.clone(), 0.5), (signal2.clone(), 0.5)], 0.75, Some(10.0));
        let fused_high_k = fuse(&[(signal1, 0.5), (signal2, 0.5)], 0.75, Some(60.0));

        assert!(fused_low_k[0].score > fused_high_k[0].score,
            "lower k should produce higher scores: {} vs {}",
            fused_low_k[0].score, fused_high_k[0].score);
    }
}
