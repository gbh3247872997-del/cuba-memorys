"""Hebbian learning module v2.0: Oja's rule + SM-2 decay + Relation learning.

Mathematical foundations:
- Oja's Rule (1982): Δw = η·(x·y - y²·w) — prevents weight explosion
- SM-2 (Wozniak 1987): Adaptive decay based on access frequency
  - Replaces fixed Ebbinghaus λ with access-count-dependent EF
  - Memories accessed 5+ times live 2.5x longer than unaccessed ones
- Collins & Loftus (1975): Spreading activation with 30% transmission per hop
- Hebb (1949): "Neurons that fire together wire together" — relation strength
"""

import math

# ─── Constants ────────────────────────────────────────────────────────

# Learning rate for Oja's rule
ETA: float = 0.05

# Base decay constant: half-life = 30 days → λ = ln(2) / 30
DECAY_LAMBDA_BASE: float = math.log(2) / 30.0  # ≈ 0.0231

# Spreading activation decay factor per hop
SPREAD_DECAY: float = 0.3

# Importance bounds
MIN_IMPORTANCE: float = 0.01
MAX_IMPORTANCE: float = 1.0

# SM-2 constants
SM2_EF_MIN: float = 1.3    # Minimum easiness factor
SM2_EF_MAX: float = 2.5    # Maximum easiness factor
SM2_EF_BASE: float = 2.5   # Starting easiness factor

# Relation learning constants
RELATION_TRAVERSE_BOOST: float = 0.05
RELATION_DECAY_LAMBDA: float = math.log(2) / 60.0  # 60-day half-life


# ─── Oja's Rule ───────────────────────────────────────────────────────

def oja_positive(importance: float) -> float:
    """Apply positive Hebbian reinforcement using Oja's rule.

    Formula: new_w = min(1.0, w + η·(1 - w²))
    The (1 - w²) term prevents explosion: as w→1, Δw→0.

    Args:
        importance: Current importance weight in [0, 1].

    Returns:
        New importance, guaranteed in [MIN_IMPORTANCE, MAX_IMPORTANCE].
    """
    delta = ETA * (1.0 - importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance + delta))


def oja_negative(importance: float) -> float:
    """Apply negative Hebbian reinforcement.

    Formula: new_w = max(0.01, w - η·(1 + w²))
    Always decreases, faster for higher weights.

    Args:
        importance: Current importance weight in [0, 1].

    Returns:
        New importance, guaranteed in [MIN_IMPORTANCE, MAX_IMPORTANCE].
    """
    delta = ETA * (1.0 + importance * importance)
    return min(MAX_IMPORTANCE, max(MIN_IMPORTANCE, importance - delta))


# ─── SM-2 Adaptive Decay ─────────────────────────────────────────────

def sm2_easiness_factor(access_count: int) -> float:
    """Calculate SM-2 easiness factor from access count.

    Maps access_count to a quality score (0-5) then computes EF.
    Higher access = higher quality = slower decay.

    SM-2 formula: EF' = EF + (0.1 - (5-q)·(0.08 + (5-q)·0.02))

    Args:
        access_count: Number of times the memory was accessed.

    Returns:
        Easiness factor in [SM2_EF_MIN, SM2_EF_MAX].
    """
    # Map access_count → quality (0-5)
    if access_count <= 0:
        quality = 1
    elif access_count <= 2:
        quality = 2
    elif access_count <= 5:
        quality = 3
    elif access_count <= 10:
        quality = 4
    else:
        quality = 5

    # SM-2 EF calculation from base
    ef = SM2_EF_BASE + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    return max(SM2_EF_MIN, min(SM2_EF_MAX, ef))


def sm2_decay(importance: float, days_elapsed: float, access_count: int) -> float:
    """Apply SM-2 adaptive decay (replaces fixed Ebbinghaus).

    The decay rate adapts based on how often the memory is accessed:
    - access_count=0: EF=1.30, half-life ≈ 39 days
    - access_count=3: EF=2.18, half-life ≈ 65 days
    - access_count=5: EF=2.50, half-life ≈ 75 days (2.5x longer)

    Formula: R(t) = R₀ · e^(-λ/EF · t)

    Args:
        importance: Current importance.
        days_elapsed: Days since last access.
        access_count: Number of times accessed.

    Returns:
        Decayed importance, minimum MIN_IMPORTANCE.
    """
    if days_elapsed <= 0:
        return importance

    ef = sm2_easiness_factor(access_count)
    adaptive_lambda = DECAY_LAMBDA_BASE / ef
    decayed = importance * math.exp(-adaptive_lambda * days_elapsed)
    return max(MIN_IMPORTANCE, decayed)


# ─── Legacy Ebbinghaus (kept for backward compat) ────────────────────

def ebbinghaus_decay(importance: float, days_elapsed: float) -> float:
    """Apply Ebbinghaus forgetting curve decay (legacy, use sm2_decay instead).

    Formula: R(t) = R₀ · e^(-λt), half-life = 30 days.

    Args:
        importance: Current importance.
        days_elapsed: Days since last access.

    Returns:
        Decayed importance, minimum MIN_IMPORTANCE.
    """
    if days_elapsed <= 0:
        return importance
    decayed = importance * math.exp(-DECAY_LAMBDA_BASE * days_elapsed)
    return max(MIN_IMPORTANCE, decayed)


# ─── Spreading Activation ────────────────────────────────────────────

def spreading_activation_boost(current_importance: float) -> float:
    """Calculate boost for a neighbor entity via spreading activation.

    Args:
        current_importance: Neighbor's current importance.

    Returns:
        New importance after boost, capped at MAX_IMPORTANCE.
    """
    boost = 0.02 * SPREAD_DECAY  # 0.006
    return min(MAX_IMPORTANCE, current_importance + boost)


# ─── Synapse Weight ───────────────────────────────────────────────────

def synapse_weight_boost(current_weight: float, max_weight: float = 5.0) -> float:
    """Boost error synapse weight with saturation.

    Formula: Δw = 0.1 · (1 - w/max_w)

    Args:
        current_weight: Current synapse weight.
        max_weight: Maximum allowed weight.

    Returns:
        New synapse weight.
    """
    delta = 0.1 * (1.0 - current_weight / max_weight)
    return min(max_weight, max(0.0, current_weight + delta))


# ─── Relation Strength Learning ──────────────────────────────────────

def relation_traverse_boost(current_strength: float) -> float:
    """Boost relation strength when traversed (Hebbian).

    "Edges that fire together strengthen together."

    Args:
        current_strength: Current relation strength in [0, 1].

    Returns:
        New strength, capped at MAX_IMPORTANCE.
    """
    return min(MAX_IMPORTANCE, current_strength + RELATION_TRAVERSE_BOOST)


def relation_decay(strength: float, days_since_traversal: float) -> float:
    """Decay relation strength (60-day half-life, slower than observations).

    Relations are structurally more stable than individual observations.

    Args:
        strength: Current relation strength.
        days_since_traversal: Days since last traversal.

    Returns:
        Decayed strength, minimum MIN_IMPORTANCE.
    """
    if days_since_traversal <= 0:
        return strength
    decayed = strength * math.exp(-RELATION_DECAY_LAMBDA * days_since_traversal)
    return max(MIN_IMPORTANCE, decayed)


# ─── Transitive Inference ────────────────────────────────────────────

def transitive_strength(
    strength_ab: float, strength_bc: float, depth: int,
) -> float:
    """Calculate inferred strength for transitive relation A→B→C.

    Formula: s(A→C) = s(A→B) × s(B→C) × 0.9^depth

    Args:
        strength_ab: Strength of first edge.
        strength_bc: Strength of second edge.
        depth: Current traversal depth.

    Returns:
        Inferred strength, always < 1.0.
    """
    return strength_ab * strength_bc * (0.9 ** depth)
