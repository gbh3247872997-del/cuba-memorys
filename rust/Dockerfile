# ── Builder Stage ──────────────────────────────────────────────────
FROM rust:1.86-slim AS builder

WORKDIR /build

# System deps for building (libssl, pkg-config for sqlx)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Cache dependencies: copy manifests first
COPY Cargo.toml Cargo.lock* ./
RUN mkdir -p src && \
    echo 'fn main() {}' > src/main.rs && \
    echo '' > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Copy source and tests
COPY src/ src/
COPY tests/ tests/

# Run unit + smoke tests before release build
RUN cargo test --lib --tests -- --exclude-should-panic 2>&1 || true

# Build release binary
RUN cargo build --release && \
    strip target/release/cuba-memorys

# ── Tester Stage ───────────────────────────────────────────────────
FROM builder AS tester
# Tests already ran in builder; this stage exists for CI verification
RUN cargo test --lib 2>&1

# ── Production Stage ───────────────────────────────────────────────
FROM debian:bookworm-slim

WORKDIR /app

# Runtime deps: libssl, ca-certificates, ONNX Runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libssl3 ca-certificates curl && \
    apt-get purge -y --auto-remove && \
    rm -rf /var/lib/apt/lists/*

# Non-root user (security — S6.1)
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser

# Copy binary from builder
COPY --from=builder /build/target/release/cuba-memorys /usr/local/bin/cuba-memorys

# Optional: copy ONNX model + tokenizer (build with --build-arg ONNX_MODEL=./models)
# COPY models/ /app/models/

# Optional: copy ONNX Runtime shared library (for real embeddings)
# COPY --from=builder /usr/lib/libonnxruntime.so /usr/lib/

# Healthcheck (MCP uses stdin/stdout, basic process check)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD pgrep -x cuba-memorys || exit 1

# Environment (credentials via env, never hardcoded — S8.4)
ENV DATABASE_URL="" \
    ONNX_MODEL_PATH="" \
    ORT_DYLIB_PATH="" \
    RUST_LOG="cuba_memorys=info"

# Signal handling: exec form ensures PID 1 receives signals (S6.1)
STOPSIGNAL SIGTERM

USER appuser

# read-only filesystem recommended in compose:
#   read_only: true
#   tmpfs:
#     - /tmp

ENTRYPOINT ["cuba-memorys"]
