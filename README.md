# 🧠 Cuba-Memorys

**Persistent memory for AI agents** — A Model Context Protocol (MCP) server that gives AI coding assistants long-term memory with a knowledge graph, Hebbian learning, and anti-hallucination grounding.

12 tools. Zero manual setup. Mathematically rigorous.

---

## Why Cuba-Memorys?

AI agents forget everything between conversations. Cuba-Memorys solves this by giving them:

- **A knowledge graph** — Entities, observations, and relations that persist across sessions
- **Error memory** — Never repeat the same mistake twice (anti-repetition guard)
- **Hebbian learning** — Memories strengthen with use and fade adaptively (SM-2 spaced repetition)
- **Anti-hallucination** — Verify claims against stored knowledge with graduated confidence scoring
- **Semantic search** — TF-IDF + pg_trgm + optional BGE embeddings for vocabulary-level matching
- **Graph intelligence** — PageRank, Louvain community detection, spreading activation

| Feature | Cuba-Memorys | Basic Memory MCPs |
|---------|:------------:|:-----------------:|
| Knowledge graph with relations | ✅ | ❌ |
| Hebbian learning (Oja's rule) | ✅ | ❌ |
| SM-2 adaptive spaced repetition | ✅ | ❌ |
| TF-IDF semantic search | ✅ | ❌ |
| Optional BGE embeddings (ONNX) | ✅ | ❌ |
| Contradiction detection | ✅ | ❌ |
| Graduated confidence scoring | ✅ | ❌ |
| PageRank entity importance | ✅ | ❌ |
| Louvain community detection | ✅ | ❌ |
| Shannon entropy (knowledge diversity) | ✅ | ❌ |
| Chi-squared concept drift detection | ✅ | ❌ |
| Error pattern detection + MTTR | ✅ | ❌ |
| Entity duplicate detection (rapidfuzz) | ✅ | ❌ |
| Observation versioning (audit trail) | ✅ | ❌ |
| Full JSON export/backup | ✅ | ❌ |
| Fuzzy search (typo-tolerant) | ✅ | ❌ |
| Spreading activation | ✅ | ❌ |
| Batch observations (10x fewer calls) | ✅ | ❌ |
| Auto-provisions its own DB | ✅ | ❌ |

---

## Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Docker** (for PostgreSQL)

### 2. Install

```bash
# Clone the repository
git clone https://github.com/lENADRO1910/cuba-memorys.git
cd cuba-memorys

# Start PostgreSQL (auto-creates database + schema)
docker compose up -d

# Install the package (core)
pip install -e .

# Optional: Install BGE embeddings for semantic search (adds ~1GB model download)
pip install -e ".[embeddings]"
```

### 3. Configure your AI editor

Add to your MCP configuration (e.g., `mcp_config.json`):

```json
{
  "mcpServers": {
    "cuba-memorys": {
      "command": "/path/to/cuba_memorys_launcher.sh",
      "disabled": false
    }
  }
}
```

Or run directly:

```bash
DATABASE_URL="postgresql://cuba:memorys2026@127.0.0.1:5488/brain" python -m cuba_memorys
```

The server auto-creates the `brain` database and all tables on first run. Zero manual SQL.

---

## The 12 Tools

### Knowledge Graph

| Tool | Description |
|------|------------|
| `brain_entity` | Create, read, update, delete knowledge entities. Types: `concept`, `project`, `technology`, `person`, `pattern`, `config`. Accessing an entity triggers spreading activation on its neighbors. |
| `brain_observe` | Attach observations to entities with **contradiction detection** (warns if new observation conflicts with existing ones). Supports `batch_add` for 10x fewer tool calls. Types: `fact`, `decision`, `lesson`, `preference`, `context`, `tool_usage`. |
| `brain_relate` | Connect entities with typed relations (`uses`, `causes`, `implements`, `depends_on`, `related_to`). **Traverse** walks the graph with Hebbian strength learning. **Infer** discovers transitive connections. |

### Search & Verification

| Tool | Description |
|------|------------|
| `brain_search` | 5-mode search: **hybrid** (TF-IDF + full-text + fuzzy + importance + freshness), **keyword**, **fuzzy** (pg_trgm), **graph**, **verify** (anti-hallucination — returns graduated confidence: `verified` / `partial` / `weak` / `unknown` with grounding metadata). |

### Error Memory

| Tool | Description |
|------|------------|
| `brain_error_report` | Report errors with context. Auto-detects patterns (≥3 similar = warning). Boosts synapse weight via Hebbian learning. |
| `brain_error_solve` | Mark an error as resolved. Cross-references similar unresolved errors. |
| `brain_error_query` | Search with **anti-repetition guard**: warns if a similar approach previously failed. |

### Sessions & Decisions

| Tool | Description |
|------|------------|
| `brain_session` | Track working sessions with goals and outcomes. |
| `brain_decision` | Record architecture decisions with context, alternatives, and rationale. |

### Memory Maintenance

| Tool | Description |
|------|------------|
| `brain_consolidate` | **decay** (SM-2 adaptive), **prune**, **merge**, **summarize**, **pagerank** (recalculate importance via graph structure), **find_duplicates** (rapidfuzz), **export** (full JSON backup), **stats**. |
| `brain_feedback` | RLHF loop: **positive** (Oja's rule boost), **negative** (decrease), **correct** (update with versioning audit trail). |
| `brain_analytics` | **summary** (counts + token estimate), **health** (staleness, Shannon entropy, diversity score, DB size), **drift** (chi-squared with scipy p-values), **communities** (Louvain detection). |

---

## Mathematical Foundations

Cuba-Memorys is built on peer-reviewed algorithms, not ad-hoc heuristics:

### SM-2 Adaptive Decay — Wozniak (1987)

```
EF = max(1.3, 1.3 + 0.24·ln(1 + access_count))
R(t) = R₀ · e^(-0.0231/EF · t)
```

Frequently accessed memories decay slower. Unlike fixed Ebbinghaus half-life, access count modulates the decay rate — memories that are used more survive longer.

### Oja's Rule (1982) — Hebbian Learning

```
Positive: Δw = η · (1 - w²)     → converges to 1.0, cannot explode
Negative: Δw = η · (1 + w²)     → converges to 0.01 (floor)
```

Where `η = 0.05`. The `w²` term provides natural saturation.

### TF-IDF Semantic Search — Salton (1975)

```
tfidf(t, d) = tf(t, d) · log(N / df(t))
similarity = cosine(tfidf(q), tfidf(d))
```

Vocabulary-level semantic matching via scikit-learn, beyond character-level pg_trgm.

### Optional BGE Embeddings — BAAI (2023)

```
model: Qdrant/bge-small-en-v1.5-onnx-Q (quantized, ~130MB)
runtime: ONNX (no PyTorch dependency)
similarity: cosine(embed(query), embed(observation))
```

Auto-downloads on first use. Falls back to TF-IDF if not installed.

### PageRank — Brin & Page (1998)

```
PR(v) = (1-α)/N + α · Σ PR(u)/deg(u)     where α = 0.85
final_importance = 0.6·PR + 0.4·current_importance
```

Graph-structural importance via networkx. Entities with more inbound relations rank higher.

### Shannon Entropy — Knowledge Diversity

```
H = -Σ p·log₂(p)
diversity_score = H / H_max
```

Measures how evenly distributed your knowledge is across entity types and observation types. Score 0–1.

### Spreading Activation — Collins & Loftus (1975)

When entity X is accessed, its graph neighbors receive a small importance boost (0.6% per hop, decaying 30% per level).

### Chi-Squared Concept Drift — Pearson (1900)

```
χ² = Σ (observed - expected)² / expected
p-value via scipy.stats.chi2.sf()
```

Detects when the distribution of error types changes significantly.

### Contradiction Detection

```
conflict = tfidf_similarity(new, existing) > 0.7 AND has_negation(new, existing)
```

Checks for semantic overlap + negation patterns (not, never, instead of, replaced, deprecated, etc.) in both English and Spanish.

---

## Architecture

```
cuba-memorys/
├── docker-compose.yml          # Dedicated PostgreSQL (port 5488)
├── pyproject.toml              # Package metadata + optional deps
├── README.md
└── src/cuba_memorys/
    ├── __init__.py
    ├── __main__.py             # Entry point
    ├── server.py               # 12 MCP tool handlers (~1700 LOC)
    ├── db.py                   # asyncpg pool + orjson + auto-DB creation
    ├── schema.sql              # 5 tables, 15 indexes, pg_trgm, versioning
    ├── hebbian.py              # SM-2, Oja's rule, spreading activation
    ├── search.py               # LRU cache, confidence scoring, contradiction detection
    ├── tfidf.py                # TF-IDF semantic search (scikit-learn)
    └── embeddings.py           # Optional BGE embeddings (ONNX Runtime)
```

### Database Schema

| Table | Purpose | Key Features |
|-------|---------|-------------|
| `brain_entities` | Knowledge graph nodes | tsvector + pg_trgm indexes, importance ∈ [0,1] |
| `brain_observations` | Facts attached to entities | 8 types, provenance tracking, versioning, FK cascade |
| `brain_relations` | Graph edges | 5 types, bidirectional, Hebbian strength learning |
| `brain_errors` | Error memory | JSONB context, synapse weight, MTTR tracking |
| `brain_sessions` | Working sessions | Goals (JSONB), outcome tracking |

### Search Pipeline

Cuba-Memorys uses **multi-signal scoring** to combine up to 5 signals:

**With embeddings available:**
| Signal | Weight |
|--------|:------:|
| BGE embedding cosine similarity | 35% |
| Full-text search (ts_rank_cd) | 20% |
| TF-IDF cosine similarity | 15% |
| Importance (Hebbian-weighted) | 15% |
| Freshness (recency decay) | 15% |

**Without embeddings (TF-IDF fallback):**
| Signal | Weight |
|--------|:------:|
| Full-text search (ts_rank_cd) | 25% |
| TF-IDF cosine similarity | 25% |
| Fuzzy matching (pg_trgm) | 20% |
| Importance | 20% |
| Freshness | 10% |

### Dependencies

**Core** (always installed):
- `asyncpg` — PostgreSQL async driver
- `orjson` — 5-10x faster JSON serialization (handles UUID/datetime)
- `scikit-learn` — TF-IDF vectorization
- `networkx` — PageRank + Louvain community detection
- `scipy` — Chi-squared statistical tests
- `rapidfuzz` — Entity duplicate detection

**Optional** (`pip install -e ".[embeddings]"`):
- `onnxruntime` — ONNX model inference
- `huggingface-hub` — Auto-download BGE model
- `tokenizers` — Fast tokenization

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |

### Docker Compose

The included `docker-compose.yml` runs a dedicated PostgreSQL 18 Alpine instance:

- **Port**: 5488 (avoids conflicts with 5432/5433)
- **Resources**: 256MB RAM, 0.5 CPU
- **Restart**: always (auto-starts on boot)
- **Healthcheck**: `pg_isready` every 10s

### Customizing the Launcher

Edit `cuba_memorys_launcher.sh` to change connection parameters:

```bash
DB_PORT="5488"      # Change if 5488 is taken
DB_USER="cuba"
DB_PASS="memorys2026"
DB_NAME="brain"
```

---

## Verification

Cuba-Memorys has been tested with a NEMESIS protocol (3-level) test suite:

```
15/15 tests passed — ZERO DEFECTS

🟢 Normal    — Entity CRUD, observe, search (hybrid + verify), relate (create + traverse),
                PageRank, communities, decay, find_duplicates, export, health + entropy
🟡 Pessimist — Nonexistent entities, unknown claims (confidence=0.0), empty batch_add
🔴 Extreme   — SQL injection, XSS payloads, 10KB content, concurrent creates,
                Unicode/emoji, mathematical precision validation
```

### Performance

| Operation | Avg latency |
|-----------|:----------:|
| Hybrid search | < 5ms |
| Analytics | < 2.5ms |
| Entity CRUD | < 1ms |
| PageRank (100 entities) | < 50ms |

---

## How It Works in Practice

### 1. The agent learns from your project

```
Agent: I learned that FastAPI endpoints must use async def with response_model.
→ brain_entity(create, "FastAPI", technology)
→ brain_observe(add, "FastAPI", "All endpoints must be async def with response_model")
```

### 2. Error memory prevents repeated mistakes

```
Agent: I got IntegrityError: duplicate key on numero_parte.
→ brain_error_report("IntegrityError", "duplicate key on numero_parte")
→ Similar error found! Solution: "Add SELECT EXISTS before INSERT with FOR UPDATE"
```

### 3. Anti-hallucination grounding

```
Agent: Let me verify this claim before responding...
→ brain_search("FastAPI uses Django ORM", mode="verify")
→ confidence: 0.0, level: "unknown"
→ recommendation: "No supporting evidence found. High hallucination risk."
```

### 4. Memories adapt with SM-2

```
Access count 0:  half-life ≈ 30 days (standard Ebbinghaus)
Access count 5:  half-life ≈ 40 days (SM-2 adaptive, decays slower)
Access count 20: half-life ≈ 60 days (frequently used, survives much longer)
```

### 5. Contradiction detection

```
Existing:  "Project uses PostgreSQL 15"
New:       "Project does NOT use PostgreSQL, it uses MySQL instead"
→ ⚠️ CONFLICT detected (similarity=0.82, negation pattern: "does NOT")
```

### 6. Graph intelligence

```
→ brain_consolidate(action="pagerank")
→ Top entities: FastAPI (0.42), SQLAlchemy (0.38), PostgreSQL (0.35)

→ brain_analytics(metric="communities")
→ Community 1: [FastAPI, Pydantic, SQLAlchemy] — Backend cluster
→ Community 2: [React, Next.js, TypeScript] — Frontend cluster
```

---

## License

[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — Free to use and modify, **not for commercial use**.

---

## Author

**Leandro Pérez G.**

- GitHub: [@lENADRO1910](https://github.com/lENADRO1910)
- Email: [leandropatodo@gmail.com](mailto:leandropatodo@gmail.com)

## Credits

Mathematical foundations: Wozniak (1987, SM-2), Oja (1982), Salton (1975, TF-IDF), Brin & Page (1998, PageRank), Collins & Loftus (1975), Shannon (1948), Pearson (1900, χ²), Blondel et al. (2008, Louvain), BAAI (2023, BGE embeddings).
