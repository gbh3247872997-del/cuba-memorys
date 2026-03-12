# 🧠 Cuba-Memorys — Rust

[![Rust](https://img.shields.io/badge/rust-1.93+-orange?logo=rust&logoColor=white)](https://rust-lang.org)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/license-CC%20BY--NC%204.0-green)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com/lENADRO1910/cuba-memorys)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-18-336791?logo=postgresql&logoColor=white)](https://postgresql.org)
[![MCP](https://img.shields.io/badge/MCP-compatible-8A2BE2)](https://modelcontextprotocol.io)
[![Tests](https://img.shields.io/badge/tests-68%20pass-brightgreen)](https://github.com/lENADRO1910/cuba-memorys)
[![Tech Debt](https://img.shields.io/badge/tech%20debt-0-brightgreen)](https://github.com/lENADRO1910/cuba-memorys)

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server written in Rust for maximum performance. Knowledge graph with neuroscience-inspired algorithms: Hebbian learning, FSRS-6 decay, Dual-Strength memory, Leiden communities, BM25+ search, and anti-hallucination grounding.

12 tools · 4.6MB binary · Sub-millisecond handlers · Zero tech debt

---

## Why Rust?

The Python implementation works. The Rust rewrite makes it **fast**:

| Metric | Python v1.6.0 | Rust v2.0.0 |
| ------ | :-----------: | :---------: |
| Binary size | ~50MB (venv) | **4.6MB** |
| Entity create | ~2ms | **498µs** |
| Entity get | ~3ms | **1.86ms** |
| Observation add | ~2ms | **474µs** |
| Hybrid search | <5ms | **2.52ms** |
| Analytics | <2.5ms | **958µs** |
| Memory usage | ~120MB | **~15MB** |
| Startup time | ~2s | **<100ms** |
| Dependencies | 12 Python packages | **0 runtime deps** |

---

## Quick Start

### 1. Prerequisites

- **Rust 1.93+** (edition 2024)
- **PostgreSQL 18** with `pg_trgm` and `vector` extensions

### 2. Build

```bash
cd cuba-memorys/rust

# Release build (4.6MB stripped binary)
cargo build --release

# Run tests (68 total: 57 unit + 11 smoke)
cargo test
```

### 3. Run

```bash
DATABASE_URL="postgresql://user:pass@localhost:5433/brain" \
  ./target/release/cuba-memorys
```

The server auto-creates the `brain` database schema on first run.

### 4. Configure your AI editor

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba-memorys",
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5433/brain"
      }
    }
  }
}
```

### 5. Optional: ONNX Embeddings

For real BGE-small-en-v1.5 semantic embeddings instead of hash-based fallback:

```bash
# Download model (~130MB)
bash scripts/download_model.sh

# Set environment
export ONNX_MODEL_PATH="$HOME/.cache/cuba-memorys/models"
export ORT_DYLIB_PATH="/path/to/libonnxruntime.so"
```

Without ONNX, the server uses deterministic hash-based embeddings — functional but without semantic understanding.

---

## Architecture

```text
rust/
├── Cargo.toml                  # Dependencies + release profile
├── Dockerfile                  # Multi-stage: Builder → Tester → Production
├── .dockerignore
├── examples/
│   └── bench_handlers.rs       # Benchmark script (6 handlers)
├── scripts/
│   └── download_model.sh       # ONNX model downloader
├── src/
│   ├── main.rs                 # Entry point (mimalloc, graceful shutdown)
│   ├── lib.rs                  # Public API
│   ├── protocol.rs             # JSON-RPC 2.0 + REM daemon (4h consolidation)
│   ├── db.rs                   # sqlx PgPool (10/1/5s) + idempotent schema
│   ├── schema.sql              # 5 tables, 15+ indexes, pg_trgm + pgvector
│   ├── constants.rs            # Tool definitions, thresholds, enums
│   ├── handlers/               # 12 MCP tool handlers
│   │   ├── mod.rs              # dispatch() router (all 12 tools)
│   │   ├── alma.rs             # Entity CRUD + Hebbian + Dual-Strength
│   │   ├── cronica.rs          # Observations (PE gating, density, dedup)
│   │   ├── faro.rs             # Hybrid search (weighted RRF, session-aware)
│   │   ├── expediente.rs       # Error search + anti-repetition guard
│   │   ├── alarma.rs           # Error reporting + pattern detection (≥3)
│   │   ├── remedio.rs          # Error resolution + cross-reference
│   │   ├── eco.rs              # RLHF feedback (Oja's rule)
│   │   ├── decreto.rs          # Architecture decisions
│   │   ├── jornada.rs          # Session management (goals, outcomes)
│   │   ├── puente.rs           # Relations (traverse, infer, blake3 dedup)
│   │   ├── vigia.rs            # Graph analytics (health, drift, bridges)
│   │   └── zafra.rs            # Consolidation (decay, prune, merge, export)
│   ├── cognitive/              # Neuroscience-inspired algorithms
│   │   ├── fsrs.rs             # FSRS-6 spaced repetition (power-law)
│   │   ├── dual_strength.rs    # Dual-Strength model (Bjork 1992)
│   │   ├── hebbian.rs          # Oja's rule + BCM metaplasticity
│   │   ├── spreading.rs        # RWR spreading activation (Collins 1975)
│   │   ├── prediction_error.rs # Adaptive prediction error gating
│   │   └── density.rs          # Shannon entropy gating
│   ├── embeddings/
│   │   ├── mod.rs
│   │   └── onnx.rs             # ONNX inference (ort v2) + hash fallback
│   ├── search/
│   │   ├── cache.rs            # TTL-LRU cache (O(1) lookup)
│   │   ├── rrf.rs              # Reciprocal Rank Fusion
│   │   ├── tfidf.rs            # BM25+ scoring (Lv & Zhai 2011)
│   │   └── confidence.rs       # Graduated confidence scoring
│   └── graph/
│       ├── centrality.rs       # Betweenness centrality
│       ├── community.rs        # Leiden algorithm (Traag 2019)
│       └── pagerank.rs         # Personalized PageRank
└── tests/
    ├── integration.rs          # 7 DB integration tests
    └── smoke_test.rs           # 11 smoke tests (no DB required)
```

**5,467 LOC** across 30 Rust source files + 1 SQL schema.

---

## Key Dependencies

| Crate | Purpose |
| ----- | ------- |
| `tokio` | Async runtime (multi-threaded) |
| `sqlx` | PostgreSQL (async, compile-time checked) |
| `pgvector` | Vector similarity search (cosine) |
| `petgraph` | Graph algorithms (PageRank, Leiden, centrality) |
| `ort` | ONNX Runtime (dynamic loading, optional) |
| `tokenizers` | HuggingFace tokenizers (BGE-small) |
| `blake3` | Cryptographic hashing (relation dedup) |
| `serde` / `serde_json` | JSON-RPC serialization |
| `lru` | O(1) LRU cache with TTL |
| `mimalloc` | Global allocator (performance) |
| `tracing` | Structured logging (JSON) |

---

## Database Schema

| Table | Purpose | Key Features |
| ----- | ------- | ------------ |
| `brain_entities` | Knowledge graph nodes | importance, FSRS stability, access_count, GIN search |
| `brain_observations` | Facts with provenance | versioning, embeddings (384d), dedup, Dual-Strength |
| `brain_relations` | Typed edges | Hebbian strength, bidirectional, blake3 hash dedup |
| `brain_errors` | Error memory | synapse weight, resolution tracking, pattern detection |
| `brain_sessions` | Working sessions | goals (JSONB), outcomes, duration tracking |

All tables use `UUIDv4` primary keys, `timestamptz` timestamps, and cascading deletes.

---

## 🇨🇺 The 12 Tools

| Tool | Meaning | Description |
| ---- | ------- | ----------- |
| `cuba_alma` | Soul | Entity CRUD with Hebbian + Dual-Strength boost |
| `cuba_cronica` | Chronicle | Observations with PE gating + Shannon density |
| `cuba_puente` | Bridge | Relations: traverse, infer, blake3 dedup |
| `cuba_faro` | Lighthouse | Weighted RRF search + session-aware boosting |
| `cuba_alarma` | Alarm | Report errors (auto-detects patterns ≥3) |
| `cuba_remedio` | Remedy | Resolve errors with cross-reference |
| `cuba_expediente` | Case file | Search errors + anti-repetition guard |
| `cuba_jornada` | Workday | Session tracking with goals/outcomes |
| `cuba_decreto` | Decree | Architecture decisions (record/query/list) |
| `cuba_zafra` | Harvest | Consolidation: decay, prune, merge, pagerank, export |
| `cuba_eco` | Echo | RLHF feedback (Oja positive/negative/correct) |
| `cuba_vigia` | Watchman | Analytics: summary, health, drift, communities, bridges |

---

## REM Sleep Daemon

A background consolidation process runs every **4 hours**, executing:

1. **FSRS decay** — Power-law forgetting with session protection
2. **PageRank** — Recalculate importance across the knowledge graph
3. **RWR Spreading Activation** — Collins & Loftus (1975) neighbor boosting
4. **TF-IDF rebuild** — BM25+ index refresh (in-memory)

Active session entities are **protected** from decay (exact `entity_id` match, not ILIKE).

---

## Benchmark

```bash
DATABASE_URL="..." cargo run --release --example bench_handlers
```

Results on local PostgreSQL (Intel i7, NVMe):

```text
  ═══ cuba-memorys Rust Benchmark ═══

  embed (hash)           0µs /call  (4000 calls)
  alma::create       498µs /call  (100 calls)
  alma::get         1.86ms /call  (100 calls)
  cronica::add       474µs /call  (100 calls)
  faro::search      2.52ms /call  (50 calls)
  vigia::summary     958µs /call  (50 calls)
```

---

## Docker

Multi-stage Dockerfile (Builder → Tester → Production):

```bash
docker build -t cuba-memorys .
docker run -e DATABASE_URL="..." cuba-memorys
```

Features:

- Non-root user (`appuser`)
- `HEALTHCHECK` via process monitor
- `STOPSIGNAL SIGTERM` (exec form, PID 1)
- Stripped binary (4.6MB)

---

## Testing

```bash
# Unit + smoke tests (no DB required)
cargo test

# Integration tests (requires PostgreSQL)
DATABASE_URL="postgresql://..." cargo test --test integration -- --ignored
```

**68 tests total**: 57 unit + 11 smoke + 7 integration (ignored without DB).

| Category | Count | Coverage |
| -------- | :---: | -------- |
| Cognitive algorithms | 21 | FSRS, Hebbian/BCM, Dual-Strength, PE gating, density |
| Search & scoring | 12 | BM25+, RRF, confidence, TF-IDF |
| Graph algorithms | 9 | Leiden (community), PageRank, centrality |
| Protocol & schema | 11 | JSON-RPC, tool definitions, schema validity |
| Smoke (no DB) | 11 | Constants, thresholds, invariants |
| Integration (DB) | 7 | Full handler round-trips (requires PostgreSQL) |

---

## Mathematical Foundations

| Algorithm | Reference | Application |
| --------- | --------- | ----------- |
| **FSRS-6** | Ye (2024) | Power-law forgetting: `R = (1 + t/9S)^(-0.5)` |
| **Dual-Strength** | Bjork & Bjork (1992) | Storage + retrieval strength separation |
| **BCM Metaplasticity** | Bienenstock, Cooper, Munro (1982) | Sliding threshold prevents runaway potentiation |
| **Oja's Rule** | Oja (1982) | Normalized Hebbian: `Δw = η·x·(y - w·x)` |
| **BM25+** | Lv & Zhai (SIGIR 2011) | Monotonic term scoring with δ=1 |
| **Weighted RRF** | Cormack (2009) | Shannon entropy-routed multi-signal fusion |
| **Leiden** | Traag, Waltman, van Eck (Nature 2019) | Community detection with connectivity guarantee |
| **PageRank** | Brin & Page (1998) | Personalized importance ranking |
| **Shannon Entropy** | Shannon (1948) | Information density gating for observations |
| **Prediction Error** | Friston (2023) | Adaptive novelty thresholds (free energy) |
| **Spreading Activation** | Collins & Loftus (1975) | Random Walk with Restart (RWR) neighbor boost |
| **BGE-small-en-v1.5** | BAAI (2023) | ONNX quantized 384d semantic embeddings |
| **blake3** | O'Connor et al. (2020) | Cryptographic hash for relation deduplication |

---

## Code Quality

Last audit: **2026-03-11** — Zero technical debt.

| Metric | Result |
| ------ | :----: |
| `cargo build --release` warnings | **0** |
| `cargo test` pass rate | **68/68 (100%)** |
| TODO/FIXME/HACK in production code | **0** |
| `panic!`/`unreachable!`/`todo!` in prod | **0** |
| Unsafe `unwrap()` in production | **0** |
| Dead code without justification | **0** |
| Hardcoded secrets | **0** |
| SQL injection vectors | **0** |
| Total LOC (src/) | **5,467** |

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

## Author

**Leandro Pérez G.**

- GitHub: [@lENADRO1910](https://github.com/lENADRO1910)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)
