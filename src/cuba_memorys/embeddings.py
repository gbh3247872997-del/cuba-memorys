"""Optional embedding module: BGE-small-en-v1.5 via ONNX Runtime.

Downloads the model automatically on first use (~80MB, one time).
If onnxruntime is not installed, all functions return empty results
and the rest of the system falls back to TF-IDF.

Model: BAAI/bge-small-en-v1.5 (Beijing Academy of AI)
- 33M parameters, 384 dimensions, MTEB 62.17
- INT8 quantized ONNX format: ~80MB disk, ~200MB RAM
- CPU only, no GPU required, ~5ms per text

References:
- Xiao et al. (2023): C-Pack: Packaged Resources for General Chinese Embeddings
- ONNX Runtime: cross-platform inference engine
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np

# State
_session: Any | None = None
_tokenizer: Any | None = None
_init_attempted: bool = False
_available: bool = False

MODEL_REPO = "Qdrant/bge-small-en-v1.5-onnx-Q"
MODEL_DIR = Path.home() / ".cache" / "cuba-memorys" / "models"
EMBEDDING_DIM = 384


def _ensure_model() -> bool:
    """Download and load model on first call. Zero human interaction.

    Returns:
        True if model is ready, False if unavailable.
    """
    global _session, _tokenizer, _init_attempted, _available

    if _init_attempted:
        return _available

    _init_attempted = True

    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError:
        print(
            "[cuba-memorys] onnxruntime not installed — "
            "embeddings disabled, using TF-IDF fallback",
            file=sys.stderr,
        )
        return False

    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="model_quantized.onnx",
            cache_dir=str(MODEL_DIR),
        )
        tokenizer_path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename="tokenizer.json",
            cache_dir=str(MODEL_DIR),
        )

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        _session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        _tokenizer = Tokenizer.from_file(tokenizer_path)
        _tokenizer.enable_truncation(max_length=512)
        _tokenizer.enable_padding(length=512)

        _available = True
        print(
            "[cuba-memorys] Embedding model loaded: BGE-small-en-v1.5 (384d)",
            file=sys.stderr,
        )
        return True

    except Exception as e:
        print(
            f"[cuba-memorys] Embedding model load failed: {e}",
            file=sys.stderr,
        )
        return False


def embed(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        texts: List of strings to embed.

    Returns:
        numpy array of shape (len(texts), 384) with L2-normalized embeddings.
        Empty array (shape (0, 384)) if model unavailable.
    """
    if not _ensure_model() or _session is None or _tokenizer is None:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    encodings = _tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array(
        [e.attention_mask for e in encodings], dtype=np.int64,
    )
    token_type_ids = np.zeros_like(input_ids)

    outputs = _session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    # Mean pooling over token embeddings (masked)
    token_embeddings = outputs[0]  # (batch, seq_len, hidden_dim)
    mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
    sum_mask = np.sum(mask_expanded, axis=1)
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
    embeddings = sum_embeddings / sum_mask

    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-9, a_max=None)
    embeddings = embeddings / norms

    return embeddings.astype(np.float32)


def cosine_sim(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity in [-1.0, 1.0]. Returns 0.0 for zero vectors.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def is_available() -> bool:
    """Check if the embedding model is loaded and ready."""
    return _available and _session is not None
