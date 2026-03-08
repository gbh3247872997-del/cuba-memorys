"""TF-IDF semantic search module: corpus-aware vectorization + cosine similarity.

Uses scikit-learn's TfidfVectorizer for vocabulary-level semantic matching.
Solves the fundamental limitation of pg_trgm: character-level similarity
cannot relate "PostgreSQL" to "database" (0% character overlap).

TF-IDF captures term frequency patterns: documents mentioning "PostgreSQL"
likely also mention "database", building implicit semantic relationships.

References:
- Salton et al. (1975): Term Frequency – Inverse Document Frequency
- scikit-learn TfidfVectorizer: L2-normalized TF-IDF with sublinear TF
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TFIDFIndex:
    """In-memory TF-IDF index for semantic search over text corpus.

    Maintains a fitted vectorizer and document matrix. Rebuilt on corpus
    changes (cache_clear). Thread-safe for read operations.
    """

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        self._matrix: np.ndarray | None = None
        self._corpus: list[str] = []
        self._fitted = False

    def fit(self, corpus: list[str]) -> None:
        """Build TF-IDF matrix from corpus.

        Args:
            corpus: List of text documents to index.
        """
        if not corpus:
            self._fitted = False
            self._matrix = None
            self._corpus = []
            return

        self._corpus = corpus
        self._matrix = self._vectorizer.fit_transform(corpus)
        self._fitted = True

    def query(self, text: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Find most similar documents to query text.

        Args:
            text: Query text.
            top_k: Maximum results to return.

        Returns:
            List of (document_index, cosine_similarity_score) tuples,
            sorted by similarity descending. Scores in [0.0, 1.0].
        """
        if not self._fitted or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        # Get top-k indices sorted by score
        if len(scores) <= top_k:
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            (int(idx), float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0.0
        ]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Uses the fitted vocabulary if available, otherwise fits on-the-fly.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Cosine similarity in [0.0, 1.0].
        """
        if self._fitted:
            vecs = self._vectorizer.transform([text_a, text_b])
        else:
            temp = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 2),
                sublinear_tf=True, strip_accents="unicode",
            )
            vecs = temp.fit_transform([text_a, text_b])

        score = cosine_similarity(vecs[0:1], vecs[1:2])[0, 0]
        return max(0.0, min(1.0, float(score)))

    def clear(self) -> None:
        """Reset the index. Must call fit() again before querying."""
        self._matrix = None
        self._corpus = []
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the index has been fitted with a corpus."""
        return self._fitted

    @property
    def corpus_size(self) -> int:
        """Number of documents in the index."""
        return len(self._corpus)


# Module-level singleton
tfidf_index = TFIDFIndex()
