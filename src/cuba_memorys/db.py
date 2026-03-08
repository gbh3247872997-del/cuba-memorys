"""Database module v2.0: asyncpg pool + orjson serialization.

Uses DATABASE_URL environment variable for connection.
Auto-creates the database AND tables on first run.
Zero manual setup required — fully autonomous.

v2.0 changes:
- orjson for 5-10x faster serialization with native UUID/datetime
- Pool tuned: min_size=2, max_size=10, statement_cache_size=512
- Semaphore increased to 8 concurrent operations
"""

import asyncio
import os
import sys
from datetime import datetime, date
from importlib import resources
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import asyncpg
import orjson

# Connection pool singleton
_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()
_semaphore = asyncio.Semaphore(8)  # v2.0: up from 3


def _get_admin_url_and_db() -> tuple[str, str]:
    """Parse DATABASE_URL into admin URL (pointing to 'postgres' db) and target db name.

    Returns:
        Tuple of (admin_url pointing to 'postgres' db, target database name).

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError(
            "DATABASE_URL environment variable is required. "
            "Example: postgresql://user:pass@localhost:5432/brain"
        )

    parsed = urlparse(url)
    target_db = parsed.path.lstrip("/") or "brain"

    # Build admin URL pointing to default 'postgres' database
    admin_parsed = parsed._replace(path="/postgres")
    admin_url = urlunparse(admin_parsed)

    return admin_url, target_db


async def _ensure_database_exists() -> None:
    """Create the target database if it doesn't exist.

    Connects to the default 'postgres' database via direct PG connection
    (not PgBouncer) to run CREATE DATABASE. Idempotent and race-safe.
    """
    admin_url, target_db = _get_admin_url_and_db()

    try:
        conn = await asyncpg.connect(admin_url, timeout=10)
    except (OSError, asyncpg.PostgresError) as e:
        print(f"[cuba-memorys] Cannot connect to PostgreSQL: {e}", file=sys.stderr)
        return

    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1",
            target_db,
        )
        if not exists:
            await conn.execute(f'CREATE DATABASE "{target_db}"')
            print(f"[cuba-memorys] Created database '{target_db}'", file=sys.stderr)
        else:
            print(f"[cuba-memorys] Database '{target_db}' OK", file=sys.stderr)
    except asyncpg.DuplicateDatabaseError:
        pass
    finally:
        await conn.close()


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool (lazy singleton).

    Auto-creates the database if it doesn't exist.
    v2.0: Tuned pool with statement_cache_size for prepared statements.

    Returns:
        asyncpg.Pool: Connection pool.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    global _pool
    if _pool is not None:
        return _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        await _ensure_database_exists()

        database_url = os.environ.get("DATABASE_URL", "")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is required. "
                "Example: postgresql://user:pass@localhost:5432/brain"
            )

        _pool = await asyncpg.create_pool(
            database_url,
            min_size=2,
            max_size=10,
            command_timeout=30,
            statement_cache_size=512,
        )
        return _pool


async def init_schema() -> None:
    """Initialize database schema (idempotent via IF NOT EXISTS).

    Creates extensions, tables, indexes, and triggers.
    Safe to call multiple times — all DDL uses IF NOT EXISTS.
    """
    pool = await get_pool()
    schema_sql = resources.files("cuba_memorys").joinpath("schema.sql").read_text()
    async with pool.acquire() as conn:
        await conn.execute(schema_sql)
    print("[cuba-memorys] Schema initialized (v2.0)", file=sys.stderr)


async def execute(query: str, *args: Any) -> str:
    """Execute a write query and return status.

    Args:
        query: SQL query with $1, $2, ... placeholders.
        *args: Query parameters.

    Returns:
        Status string from the query execution.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)


async def fetchrow(query: str, *args: Any) -> dict[str, Any] | None:
    """Fetch a single row as dict.

    Args:
        query: SQL query.
        *args: Query parameters.

    Returns:
        Dict with column names as keys, or None if no row found.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, *args)
            return dict(row) if row else None


async def fetch(query: str, *args: Any) -> list[dict[str, Any]]:
    """Fetch multiple rows as list of dicts.

    Args:
        query: SQL query.
        *args: Query parameters.

    Returns:
        List of dicts with column names as keys.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]


async def fetchval(query: str, *args: Any) -> Any:
    """Fetch a single value.

    Args:
        query: SQL query.
        *args: Query parameters.

    Returns:
        The first column of the first row.
    """
    async with _semaphore:
        pool = await get_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)


async def close() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def _json_default(obj: Any) -> Any:
    """Handle types orjson doesn't natively serialize.

    asyncpg returns pgproto.UUID which inherits uuid.UUID but
    orjson doesn't recognize it. Convert to string.
    """
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type is not JSON serializable: {type(obj)}")


def serialize(obj: Any) -> str:
    """JSON serialize with orjson (5-10x faster, native UUID/datetime).

    Handles asyncpg.pgproto.UUID via default function.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string.
    """
    return orjson.dumps(
        obj,
        default=_json_default,
        option=orjson.OPT_NON_STR_KEYS | orjson.OPT_NAIVE_UTC,
    ).decode("utf-8")
