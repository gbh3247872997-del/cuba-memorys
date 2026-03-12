//! Custom BM25 inverted index — replaces scikit-learn TfidfVectorizer.
//!
//! FIX B7: No sklearn → runs in pure Rust, no event loop blocking.
//! Uses BM25+ (Lv & Zhai, SIGIR 2011) for term scoring.
//! BM25+ adds δ=1.0 to guarantee docs with match always outrank those without.

use std::collections::HashMap;

const BM25_K1: f64 = 1.5;
const BM25_B: f64 = 0.75;
/// BM25+ delta — ensures minimum positive contribution per matching term.
/// Lv & Zhai (2011): "When Documents Are Very Long, BM25 Fails!"
const BM25_DELTA: f64 = 1.0;
const MAX_CORPUS: usize = 10_000;

/// In-memory BM25 inverted index.
pub struct Bm25Index {
    /// Term → (doc_indices, term_frequencies).
    inverted: HashMap<String, Vec<(usize, f64)>>,
    /// Document lengths.
    doc_lengths: Vec<usize>,
    /// Average document length.
    avg_dl: f64,
    /// Total documents.
    n_docs: usize,
    /// Corpus texts (for result retrieval).
    corpus: Vec<String>,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            inverted: HashMap::new(),
            doc_lengths: Vec::new(),
            avg_dl: 0.0,
            n_docs: 0,
            corpus: Vec::new(),
        }
    }

    /// Build index from corpus. Caps at MAX_CORPUS.
    pub fn fit(&mut self, corpus: Vec<String>) {
        self.inverted.clear();
        self.doc_lengths.clear();

        // Cap corpus size (prevent unbounded memory — TD8 equivalent)
        let corpus = if corpus.len() > MAX_CORPUS {
            corpus[corpus.len() - MAX_CORPUS..].to_vec()
        } else {
            corpus
        };

        self.n_docs = corpus.len();
        if self.n_docs == 0 {
            self.avg_dl = 0.0;
            self.corpus = vec![];
            return;
        }

        let mut total_length = 0usize;

        for (doc_idx, text) in corpus.iter().enumerate() {
            let tokens = tokenize(text);
            let dl = tokens.len();
            self.doc_lengths.push(dl);
            total_length += dl;

            // Count term frequencies per document
            let mut tf_map: HashMap<&str, usize> = HashMap::new();
            for token in &tokens {
                *tf_map.entry(token.as_str()).or_default() += 1;
            }

            for (term, count) in tf_map {
                let tf = count as f64 / dl.max(1) as f64;
                self.inverted
                    .entry(term.to_string())
                    .or_default()
                    .push((doc_idx, tf));
            }
        }

        self.avg_dl = total_length as f64 / self.n_docs as f64;
        self.corpus = corpus;
    }

    /// Query index for top_k results. Returns (doc_index, bm25_score).
    pub fn query(&self, text: &str, top_k: usize) -> Vec<(usize, f64)> {
        if self.n_docs == 0 {
            return vec![];
        }

        let query_tokens = tokenize(text);
        let mut scores: HashMap<usize, f64> = HashMap::new();

        for token in &query_tokens {
            if let Some(postings) = self.inverted.get(token.as_str()) {
                // IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                let df = postings.len() as f64;
                let idf = ((self.n_docs as f64 - df + 0.5) / (df + 0.5) + 1.0).ln();

                for &(doc_idx, tf) in postings {
                    let dl = self.doc_lengths[doc_idx] as f64;
                    // BM25 score: IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgdl))
                    let numerator = tf * (BM25_K1 + 1.0);
                    let denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * dl / self.avg_dl);
                    // BM25+ (Lv & Zhai 2011): add δ to guarantee positive contribution
                    *scores.entry(doc_idx).or_default() += idf * (numerator / denominator + BM25_DELTA);
                }
            }
        }

        let mut ranked: Vec<(usize, f64)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(top_k);
        ranked
    }

    /// Get corpus text by index.
    pub fn get_text(&self, idx: usize) -> Option<&str> {
        self.corpus.get(idx).map(|s| s.as_str())
    }

    /// Check if index is fitted.
    pub fn is_fitted(&self) -> bool {
        self.n_docs > 0
    }
}

/// Simple whitespace + lowercase tokenizer with basic cleaning.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|s| s.len() > 1) // Skip single chars
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let mut idx = Bm25Index::new();
        idx.fit(vec![
            "rust is a systems programming language".to_string(),
            "python is great for data science".to_string(),
            "javascript runs in the browser".to_string(),
        ]);

        let results = idx.query("rust programming", 3);
        assert!(!results.is_empty(), "should find results");
        assert_eq!(results[0].0, 0, "first result should be doc 0 (rust)");
    }

    #[test]
    fn test_bm25_empty() {
        let mut idx = Bm25Index::new();
        idx.fit(vec![]);
        assert!(idx.query("hello", 5).is_empty());
    }

    #[test]
    fn test_bm25_exact_match_scores_higher() {
        let mut idx = Bm25Index::new();
        idx.fit(vec![
            "database migration alembic postgresql".to_string(),
            "frontend react component state".to_string(),
            "postgresql database indexing performance".to_string(),
        ]);

        let results = idx.query("postgresql database", 3);
        assert!(results.len() >= 2);
        // Both docs with "postgresql" and "database" should score higher
        let top_docs: Vec<usize> = results.iter().map(|r| r.0).collect();
        assert!(top_docs.contains(&0));
        assert!(top_docs.contains(&2));
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello World! (test-123)");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"123".to_string()));
    }
}
