"""Hybrid search module v2.0: TF-IDF + embeddings + RRF scoring.

Scoring v2.0 (with embeddings available):
    score = 0.35·embedding + 0.20·ts_rank_cd + 0.15·tfidf + 0.15·importance + 0.15·freshness

Scoring v2.0 (TF-IDF fallback — no embeddings):
    score = 0.25·ts_rank_cd + 0.25·tfidf + 0.20·pg_trgm + 0.20·importance + 0.10·freshness

Verify mode v2.0: graduated confidence score [0.0, 1.0] with 4 levels.

Contradiction detection: TF-IDF similarity + negation patterns.

References:
- Robertson (2009): Reciprocal Rank Fusion
- Salton (1975): TF-IDF
- Wozniak (1987): SM-2 spaced repetition
"""

import re
import time
from typing import Any

# LRU Cache for repeated queries (TTL = 60 seconds)
_cache: dict[int, tuple[float, list[dict[str, Any]]]] = {}
CACHE_TTL: float = 60.0
CACHE_MAX_SIZE: int = 100

# Negation patterns for contradiction detection
_NEGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bno\b|\bnot\b|\bnever\b|\bya no\b|\bno es\b", re.IGNORECASE),
    re.compile(r"\bcambió\s+de\b|\bchanged\s+from\b|\breplaced\b", re.IGNORECASE),
    re.compile(r"\ben vez de\b|\binstead of\b|\brather than\b", re.IGNORECASE),
    re.compile(r"\bremoved\b|\belimina\b|\bdeprecated\b|\bobsolete\b", re.IGNORECASE),
    re.compile(r"\bwas\b.*\bnow\b|\bantes\b.*\bahora\b", re.IGNORECASE),
]


def _cache_key(query: str, mode: str, scope: str, limit: int) -> int:
    """Generate cache key from query parameters (v2.0: built-in hash, no MD5)."""
    return hash((query, mode, scope, limit))


def cache_get(query: str, mode: str, scope: str, limit: int) -> list[dict[str, Any]] | None:
    """Get cached results if still valid.

    Args:
        query: Search query.
        mode: Search mode.
        scope: Search scope.
        limit: Result limit.

    Returns:
        Cached results or None if expired/missing.
    """
    key = _cache_key(query, mode, scope, limit)
    if key in _cache:
        ts, results = _cache[key]
        if time.monotonic() - ts < CACHE_TTL:
            return results
        del _cache[key]
    return None


def cache_set(
    query: str, mode: str, scope: str, limit: int,
    results: list[dict[str, Any]],
) -> None:
    """Store results in cache.

    Args:
        query: Search query.
        mode: Search mode.
        scope: Search scope.
        limit: Result limit.
        results: Results to cache.
    """
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest_key]

    key = _cache_key(query, mode, scope, limit)
    _cache[key] = (time.monotonic(), results)


def cache_clear() -> None:
    """Clear all cached results."""
    _cache.clear()


def has_negation(text_a: str, text_b: str) -> bool:
    """Detect negation patterns between two texts.

    Args:
        text_a: First text (typically new observation).
        text_b: Second text (existing observation).

    Returns:
        True if negation patterns are detected.
    """
    combined = f"{text_a} {text_b}"
    return any(pattern.search(combined) for pattern in _NEGATION_PATTERNS)


def compute_confidence(
    trgm_score: float,
    tfidf_score: float,
    importance: float,
    freshness_days: float,
    embedding_score: float | None = None,
) -> tuple[float, str]:
    """Compute graduated confidence score for verify mode.

    Args:
        trgm_score: pg_trgm similarity in [0, 1].
        tfidf_score: TF-IDF cosine similarity in [0, 1].
        importance: Observation importance in [0, 1].
        freshness_days: Days since last access.
        embedding_score: Optional embedding cosine similarity in [0, 1].

    Returns:
        Tuple of (confidence_score, confidence_level).
        Score in [0.0, 1.0]. Level: verified/partial/weak/unknown.
    """
    freshness = 1.0 / (1.0 + freshness_days / 30.0)

    if embedding_score is not None:
        score = (
            0.30 * embedding_score
            + 0.25 * trgm_score
            + 0.20 * tfidf_score
            + 0.15 * importance
            + 0.10 * freshness
        )
    else:
        score = (
            0.30 * trgm_score
            + 0.25 * tfidf_score
            + 0.20 * (trgm_score * tfidf_score) ** 0.5  # geometric mean
            + 0.15 * importance
            + 0.10 * freshness
        )

    score = max(0.0, min(1.0, score))

    if score >= 0.8:
        level = "verified"
    elif score >= 0.5:
        level = "partial"
    elif score >= 0.3:
        level = "weak"
    else:
        level = "unknown"

    return round(score, 4), level


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text (≈ chars / 4).

    Args:
        text: Text to estimate.

    Returns:
        Estimated token count.
    """
    return max(1, len(text) // 4)


# ─── SQL Templates ────────────────────────────────────────────────────

SEARCH_ENTITIES_SQL = """
SELECT id, name, entity_type, importance, access_count,
    created_at, updated_at,
    (
        0.35 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) +
        0.30 * similarity(name, $1) +
        0.25 * importance +
        0.10 * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - updated_at)) / 86400.0))
    ) AS score
FROM brain_entities
WHERE search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(name, $1) > 0.3
ORDER BY score DESC
LIMIT $2
"""

SEARCH_OBSERVATIONS_SQL = """
SELECT o.id, o.content, o.observation_type, o.importance,
    o.source, o.created_at, o.last_accessed, o.access_count,
    e.name AS entity_name, e.entity_type,
    (
        0.35 * ts_rank_cd(o.search_vector, plainto_tsquery('simple', $1)) +
        0.30 * similarity(o.content, $1) +
        0.25 * o.importance +
        0.10 * (1.0 / (1.0 + EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0))
    ) AS score,
    similarity(o.content, $1) AS trgm_similarity,
    EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0 AS days_since_access
FROM brain_observations o
JOIN brain_entities e ON o.entity_id = e.id
WHERE o.search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(o.content, $1) > 0.3
ORDER BY score DESC
LIMIT $2
"""

SEARCH_ERRORS_SQL = """
SELECT id, error_type, error_message, solution, resolved,
    synapse_weight, project, created_at,
    (
        0.40 * ts_rank_cd(search_vector, plainto_tsquery('simple', $1)) +
        0.30 * similarity(error_message, $1) +
        0.30 * (synapse_weight / GREATEST(
            (SELECT MAX(synapse_weight) FROM brain_errors), 1.0))
    ) AS score
FROM brain_errors
WHERE search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(error_message, $1) > 0.3
ORDER BY score DESC
LIMIT $2
"""

VERIFY_SQL = """
SELECT
    o.content,
    o.importance,
    o.access_count,
    similarity(o.content, $1) AS trgm_similarity,
    ts_rank_cd(o.search_vector, plainto_tsquery('simple', $1)) AS ts_rank,
    EXTRACT(EPOCH FROM (NOW() - o.last_accessed)) / 86400.0 AS days_since_access,
    e.name AS entity_name
FROM brain_observations o
JOIN brain_entities e ON o.entity_id = e.id
WHERE o.search_vector @@ plainto_tsquery('simple', $1)
   OR similarity(o.content, $1) > 0.3
ORDER BY similarity(o.content, $1) DESC
LIMIT 5
"""
