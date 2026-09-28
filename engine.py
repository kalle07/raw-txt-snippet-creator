from __future__ import annotations

"""Persistent positional text index and search engine.

Dependencies:
    lancedb, numpy, pyarrow, rapidfuzz

Public API is intentionally compatible with the existing GUI:
    TextSearchEngine.search_with_report()
    TextSearchEngine.search()
    TextSearchEngine.index_files()
    TextSearchEngine.sync_search_folder()
    TextSearchEngine.list_documents()
    SearchHit / SearchReport / format_results

Index layout
------------
DOCS
    One row per source document.

VOCABULARY
    One row per distinct normalized token.  ``term_id`` is the numeric key
    used everywhere else in the persistent index.

POSTINGS
    One row per ``(doc_id, term_id)``.  ``positions`` is a flat Arrow
    ``list<int32>`` containing ``[start1, length1, start2, length2, ...]``.
    End offsets are reconstructed as ``start + vocabulary.token_length``.

TERM_FREQUENCIES
    One row per ``(doc_id, term_id)`` with only the occurrence count.  This
    lets the planner reject documents before positional arrays are fetched.

FTS_TOKENS
    One row per ``(doc_id, normalized_token)`` and indexed with LanceDB/Tantivy
    on the single-token ``token`` column using the ``raw`` tokenizer.  This is
    only a candidate-document accelerator.  The positional index remains the
    authoritative source for final matching and proximity checks.

The search pipeline is deliberately a funnel:
    Tantivy candidate docs
        -> candidate term/variant resolution
        -> lightweight frequencies from LanceDB
        -> numeric positional postings from LanceDB
        -> NumPy proximity engine
        -> result documents
"""

import hashlib
import inspect
import logging
import re
import shutil
import time
import traceback
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

import lancedb
import numpy as np
import pyarrow as pa
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DB_PATH = "text_search.lancedb"
SEARCH_FOLDER = "txt"
FUZZY_DISTANCE = 1
MAX_DISTANCE_CHARS = 200
SNIPPET_CONTEXT_CHARS = 200
MAX_RESULTS: int | None = 20
MAX_PROXIMITY_CHAINS_PER_DOCUMENT: int | None = None

FTS_TABLE = "fts_tokens_v3"
FTS_TOKEN_COLUMN = "token"
SEARCH_DEBUG_LOG = "search_debug.log"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _file_handler = logging.FileHandler(SEARCH_DEBUG_LOG, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_file_handler)

SearchMode = Literal["regex", "fuzzy"]
ProgressCallback = Callable[[str, float], None]


def _notify_progress(
    callback: ProgressCallback | None,
    stage: str,
    fraction: float,
) -> None:
    """Report progress without allowing UI callback failures to affect the engine."""
    if callback is None:
        return
    try:
        callback(stage, max(0.0, min(1.0, float(fraction))))
    except Exception:
        pass


# Regex/wildcard semantics are intentionally kept compatible with the existing
# engine. Matching is always against complete normalized indexed tokens.
TOKEN_PATTERN = re.compile(r"\b[\wÄÖÜäöüß]+\b", re.UNICODE)

# -----------------------------------------------------------------------------
# Persistent schemas
# -----------------------------------------------------------------------------

DOCS_TABLE = "docs"
POSTINGS_TABLE = "postings_v2"
LEGACY_POSTINGS_TABLES = ("postings",)
VOCABULARY_TABLE = "vocabulary"
FREQUENCIES_TABLE = "term_frequencies"

DOCS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64()),
        pa.field("path", pa.string()),
        pa.field("sha256", pa.string()),
        pa.field("content", pa.string()),
        pa.field("word_count", pa.int64()),
    ]
)

VOCABULARY_SCHEMA = pa.schema(
    [
        pa.field("term_id", pa.int32()),
        pa.field("term", pa.string()),
        pa.field("token_length", pa.int64()),
    ]
)

POSTINGS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int64()),
        pa.field("term_id", pa.int32()),
        # Flat pairs of (start, length).  End is always start + vocabulary length.
        pa.field("positions", pa.list_(pa.int32())),
    ]
)

FREQUENCIES_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int64()),
        pa.field("term_id", pa.int32()),
        pa.field("occurrence_count", pa.int64()),
    ]
)

# FTS needs the original token text, while the rest of the index stays numeric.
# A document may have many FTS rows; each row represents one distinct token.
FTS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int64()),
        pa.field("term_id", pa.int32()),
        pa.field(FTS_TOKEN_COLUMN, pa.string()),
    ]
)


# -----------------------------------------------------------------------------
# Result/config objects
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SearchConfig:
    fuzzy_distance: int = FUZZY_DISTANCE
    max_distance_chars: int = MAX_DISTANCE_CHARS
    snippet_context_chars: int = SNIPPET_CONTEXT_CHARS
    max_proximity_chains_per_document: int | None = MAX_PROXIMITY_CHAINS_PER_DOCUMENT

    def __post_init__(self) -> None:
        if self.fuzzy_distance < 0:
            raise ValueError("FUZZY_DISTANCE must be >= 0")
        if self.max_distance_chars < 0:
            raise ValueError("MAX_DISTANCE_CHARS must be >= 0")
        if self.snippet_context_chars < 0:
            raise ValueError("SNIPPET_CONTEXT_CHARS must be >= 0")
        if (
            self.max_proximity_chains_per_document is not None
            and self.max_proximity_chains_per_document < 1
        ):
            raise ValueError(
                "MAX_PROXIMITY_CHAINS_PER_DOCUMENT must be >= 1 or None"
            )


@dataclass(slots=True)
class SearchHit:
    doc_id: int
    document: dict
    matches_by_term: list[list[dict]]
    chains: list[list[dict]]
    mode: SearchMode


_ACTIVE_DB_QUERY_REPORT: ContextVar["SearchReport | None"] = ContextVar(
    "active_db_query_report", default=None
)
_DB_CACHE_SEEN_TABLES: set[str] = set()


def _table_cache_key(table: object) -> str:
    for attr in ("name", "_name", "table_name", "_table_name"):
        value = getattr(table, attr, None)
        if value is not None:
            try:
                return str(value)
            except Exception:
                pass
    return f"{type(table).__name__}@{id(table):x}"


def _best_effort_fragment_file_count(value: object) -> int | None:
    """Return a fragment/file count only when exposed by runtime metadata."""
    keys = (
        "fragments", "fragment_count", "num_fragments",
        "files", "file_count", "num_files",
        "files_touched", "fragments_touched",
    )
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                candidate = value[key]
                if isinstance(candidate, bool):
                    continue
                try:
                    if isinstance(candidate, (int, float)):
                        return int(candidate)
                    if isinstance(candidate, (list, tuple, set, frozenset)):
                        return len(candidate)
                except Exception:
                    continue
        for nested_key in ("stats", "metrics", "scan_stats", "execution_stats", "metadata"):
            nested = value.get(nested_key)
            count = _best_effort_fragment_file_count(nested)
            if count is not None:
                return count
    return None


def _query_fragment_file_count(*objects: object) -> int | None:
    for obj in objects:
        if obj is None:
            continue
        for attr in ("stats", "metrics", "scan_stats", "execution_stats", "_stats", "_metrics"):
            try:
                value = getattr(obj, attr, None)
            except Exception:
                continue
            count = _best_effort_fragment_file_count(value)
            if count is not None:
                return count
    return None


def _fts_to_arrow_without_score(
    query: object,
    limit: int,
) -> tuple[pa.Table, bool]:
    """Execute an FTS query without exposing _score to the application.

    Prefer the scanner-level ``disable_scoring_autoprojection`` option when
    the installed Python API forwards it through ``to_arrow``. Some LanceDB
    releases do not expose that option on the high-level query builder; in
    those versions, explicitly projecting _score and dropping it immediately
    after materialization is the portable warning-free fallback.
    """
    query = query.limit(max(int(limit), 1))

    try:
        signature = inspect.signature(query.to_arrow)
    except (TypeError, ValueError, AttributeError):
        signature = None

    supports_flag = bool(
        signature
        and (
            "disable_scoring_autoprojection" in signature.parameters
            or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        )
    )

    # A few LanceDB versions expose the flag directly on the query builder.
    # Use it when available, but do not rely on it as the only mechanism: the
    # warning is emitted by the lower-level Lance scanner.
    disable = getattr(query, "disable_scoring_autoprojection", None)
    if callable(disable):
        try:
            result = disable()
            if result is not None:
                query = result
        except Exception:
            pass

    query = query.select(["doc_id"])

    if supports_flag:
        try:
            return query.to_arrow(disable_scoring_autoprojection=True), False
        except TypeError:
            pass

    # Compatibility fallback for SDK versions where the scanner option is not
    # exposed by the high-level query object. The score is not used by the
    # application and is removed immediately after Lance materializes the
    # result. This keeps the warning silent and the application schema clean.
    score_query = query.select(["doc_id", "_score"])
    arrow = score_query.to_arrow()
    if "_score" in getattr(arrow, "column_names", []):
        arrow = arrow.drop(["_score"])
    return arrow, True


def _record_predicate_construction(elapsed_ms: float) -> None:
    report = _ACTIVE_DB_QUERY_REPORT.get()
    if report is not None:
        report.db_predicate_construction_ms += float(elapsed_ms)


@dataclass(slots=True)
class SearchReport:
    """Detailed timing and candidate statistics for one search execution."""

    mode: SearchMode
    total_ms: float = 0.0
    variant_resolution_ms: float = 0.0
    # Fuzzy variant-resolution breakdown. These are nested within
    # ``variant_resolution_ms`` and are therefore not included separately
    # in ``accounted_ms``.
    variant_candidate_query_ms: float = 0.0
    variant_candidate_prepare_ms: float = 0.0
    variant_rapidfuzz_ms: float = 0.0
    variant_candidate_query_rows: int = 0
    variant_candidate_term_ids: int = 0
    variant_candidate_min_length: int | None = None
    variant_candidate_max_length: int | None = None
    variant_resolution: list[dict] = field(default_factory=list)
    tantivy_ms: float = 0.0
    tantivy_by_term: list[dict] = field(default_factory=list)
    candidate_intersection_ms: float = 0.0
    candidate_docs_before_positions: int = 0
    frequency_ms: float = 0.0
    frequency_rows: int = 0
    frequency_docs: int = 0
    frequency_candidate_docs: int = 0
    anchor_selection_ms: float = 0.0
    anchor_by_doc: list[dict] = field(default_factory=list)
    postings_ms: float = 0.0
    postings_terms: int = 0
    postings_rows: int = 0
    match_build_ms: float = 0.0
    match_build_by_term: list[dict] = field(default_factory=list)
    authoritative_intersection_ms: float = 0.0
    candidate_docs_after_postings: int = 0
    proximity_ms: float = 0.0
    proximity_docs_checked: int = 0
    proximity_docs_with_hits: int = 0
    proximity_max_doc_ms: float = 0.0
    documents_ms: float = 0.0
    result_build_ms: float = 0.0
    final_documents: int = 0

    # LanceDB/PyArrow query instrumentation. Nested in existing search stages.
    db_predicate_construction_ms: float = 0.0
    db_query_setup_ms: float = 0.0
    db_execution_ms: float = 0.0
    db_to_pylist_ms: float = 0.0
    db_python_object_creation_ms: float = 0.0
    db_queries: int = 0
    db_rows_returned: int = 0
    db_bytes_returned: int = 0
    db_cache_cold_queries: int = 0
    db_cache_warm_queries: int = 0
    db_fragment_files_touched: int | None = None
    db_fragment_file_counts_known: int = 0
    db_query_breakdown: list[dict] = field(default_factory=list)
    runtime_posting_lookups: int = 0
    runtime_posting_hits: int = 0
    runtime_frequency_lookups: int = 0
    runtime_frequency_hits: int = 0
    runtime_candidate_terms: int = 0
    runtime_source_file_reads: int = 0
    runtime_lancedb_document_fallbacks: int = 0
    diagnostics: list[str] = field(default_factory=list)

    @property
    def accounted_ms(self) -> float:
        return sum(
            (
                self.variant_resolution_ms,
                self.tantivy_ms,
                self.candidate_intersection_ms,
                self.frequency_ms,
                self.anchor_selection_ms,
                self.postings_ms,
                self.match_build_ms,
                self.authoritative_intersection_ms,
                self.proximity_ms,
                self.documents_ms,
                self.result_build_ms,
            )
        )


@dataclass(slots=True)
class MatchArrays:
    """Compact in-memory occurrence representation for one query term/doc."""

    starts: np.ndarray
    ends: np.ndarray
    term_ids: np.ndarray
    edit_distances: np.ndarray

    @classmethod
    def empty(cls) -> "MatchArrays":
        return cls(
            starts=np.empty(0, dtype=np.int32),
            ends=np.empty(0, dtype=np.int32),
            term_ids=np.empty(0, dtype=np.int32),
            edit_distances=np.empty(0, dtype=np.int32),
        )

    def __len__(self) -> int:
        return int(self.starts.size)


# -----------------------------------------------------------------------------
# Generic LanceDB helpers
# -----------------------------------------------------------------------------

def table_exists(db: object, name: str) -> bool:
    try:
        listed = db.list_tables()
        names = getattr(listed, "tables", listed)
        return name in names
    except Exception:
        try:
            db.open_table(name)
            return True
        except Exception:
            return False


def schema_matches(table: object, schema: pa.Schema) -> bool:
    try:
        actual = {field.name: field.type for field in table.schema}
    except Exception:
        return False
    return all(actual.get(field.name) == field.type for field in schema)


def index_schema_is_valid(db: object) -> bool:
    required = (
        (DOCS_TABLE, DOCS_SCHEMA),
        (POSTINGS_TABLE, POSTINGS_SCHEMA),
        (VOCABULARY_TABLE, VOCABULARY_SCHEMA),
        (FREQUENCIES_TABLE, FREQUENCIES_SCHEMA),
        (FTS_TABLE, FTS_SCHEMA),
    )
    return all(
        table_exists(db, name) and schema_matches(db.open_table(name), schema)
        for name, schema in required
    )


def rows_to_arrow(rows: Sequence[dict], schema: pa.Schema) -> pa.Table:
    """Convert rows to an explicitly typed Arrow table.

    In particular, positions stay as numeric ``list<int32>`` values. NumPy
    arrays are used to construct those values rather than Python tuple trees.
    """

    if not rows:
        return pa.Table.from_pylist([], schema=schema)

    arrays = [
        pa.array([row[field.name] for row in rows], type=field.type)
        for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def add_rows(table: object, rows: Sequence[dict], schema: pa.Schema) -> None:
    if rows:
        table.add(rows_to_arrow(rows, schema))


def _record_lancedb_scalar_query(
    *,
    table: object,
    operation: str,
    execution_ms: float,
) -> None:
    report = _ACTIVE_DB_QUERY_REPORT.get()
    if report is None:
        return
    table_key = _table_cache_key(table)
    cache_state = "warm" if table_key in _DB_CACHE_SEEN_TABLES else "cold"
    _DB_CACHE_SEEN_TABLES.add(table_key)
    report.db_execution_ms += float(execution_ms)
    report.db_queries += 1
    if cache_state == "cold":
        report.db_cache_cold_queries += 1
    else:
        report.db_cache_warm_queries += 1
    fragment_file_count = _query_fragment_file_count(table)
    if fragment_file_count is not None:
        report.db_fragment_files_touched = (
            int(fragment_file_count)
            if report.db_fragment_files_touched is None
            else report.db_fragment_files_touched + int(fragment_file_count)
        )
        report.db_fragment_file_counts_known += 1
    report.db_query_breakdown.append({
        "operation": operation,
        "table": table_key,
        "cache_state": cache_state,
        "setup_ms": 0.0,
        "execution_ms": float(execution_ms),
        "to_pylist_ms": 0.0,
        "python_object_creation_ms": 0.0,
        "rows": 0,
        "bytes": 0,
        "fragment_files_touched": fragment_file_count,
    })


def query_rows(
    table: object,
    predicate: str | None = None,
    columns: list[str] | None = None,
    operation: str = "LanceDB query",
) -> list[dict]:
    setup_started = time.perf_counter()
    query = table.search()
    if predicate:
        query = query.where(predicate)
    if columns:
        query = query.select(columns)
    setup_ms = (time.perf_counter() - setup_started) * 1000.0

    started = time.perf_counter()
    arrow = query.to_arrow()
    execution_ms = (time.perf_counter() - started) * 1000.0
    arrow_bytes = int(getattr(arrow, "nbytes", 0) or 0)

    started = time.perf_counter()
    rows = arrow.to_pylist()
    to_pylist_ms = (time.perf_counter() - started) * 1000.0

    table_key = _table_cache_key(table)
    cache_state = "warm" if table_key in _DB_CACHE_SEEN_TABLES else "cold"
    _DB_CACHE_SEEN_TABLES.add(table_key)
    fragment_file_count = _query_fragment_file_count(query, arrow, table)

    report = _ACTIVE_DB_QUERY_REPORT.get()
    if report is not None:
        report.db_query_setup_ms += setup_ms
        report.db_execution_ms += execution_ms
        report.db_to_pylist_ms += to_pylist_ms
        # Inclusive: PyArrow creates the Python row objects inside to_pylist().
        report.db_python_object_creation_ms += to_pylist_ms
        report.db_queries += 1
        report.db_rows_returned += len(rows)
        report.db_bytes_returned += arrow_bytes
        if cache_state == "cold":
            report.db_cache_cold_queries += 1
        else:
            report.db_cache_warm_queries += 1
        if fragment_file_count is not None:
            report.db_fragment_files_touched = (
                int(fragment_file_count)
                if report.db_fragment_files_touched is None
                else report.db_fragment_files_touched + int(fragment_file_count)
            )
            report.db_fragment_file_counts_known += 1
        report.db_query_breakdown.append({
            "operation": operation,
            "table": table_key,
            "cache_state": cache_state,
            "setup_ms": setup_ms,
            "execution_ms": execution_ms,
            "to_pylist_ms": to_pylist_ms,
            "python_object_creation_ms": to_pylist_ms,
            "rows": len(rows),
            "bytes": arrow_bytes,
            "fragment_files_touched": fragment_file_count,
        })
    return rows


def where_in(field: str, values: Sequence[str | int]) -> str:
    started = time.perf_counter()
    if not values:
        result = "1 = 0"
    else:
        encoded = [
            str(value) if isinstance(value, (int, np.integer)) else sql_quote(str(value))
            for value in values
        ]
        result = f"{field} IN ({','.join(encoded)})"
    _record_predicate_construction((time.perf_counter() - started) * 1000.0)
    return result


def query_rows_in(
    table: object,
    field: str,
    values: Sequence[str | int],
    columns: list[str],
    chunk_size: int = 500,
    extra_predicate: str | None = None,
    operation: str = "LanceDB query",
) -> list[dict]:
    rows: list[dict] = []
    values_list = list(values)
    for offset in range(0, len(values_list), chunk_size):
        chunk = values_list[offset : offset + chunk_size]
        predicate = where_in(field, chunk)
        if extra_predicate:
            predicate += f" AND ({extra_predicate})"
        rows.extend(query_rows(table, predicate, columns, operation=operation))
    return rows


def delete_rows_in(
    table: object,
    field: str,
    values: Sequence[str | int],
    chunk_size: int = 500,
) -> None:
    values_list = list(values)
    for offset in range(0, len(values_list), chunk_size):
        chunk = values_list[offset : offset + chunk_size]
        if chunk:
            table.delete(where_in(field, chunk))


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


# -----------------------------------------------------------------------------
# Tokenization / index construction
# -----------------------------------------------------------------------------

def normalize_token(token: str) -> str:
    return token.casefold()


def iter_tokens(content: str):
    for match in TOKEN_PATTERN.finditer(content):
        yield normalize_token(match.group(0)), match.start(), match.end()


def count_words(content: str) -> int:
    """Count words using the same tokenizer as the search index."""

    return sum(1 for _ in TOKEN_PATTERN.finditer(content))


def read_text_with_sha256(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="ignore"), hashlib.sha256(raw).hexdigest()


def document_row(doc_id: int, path: Path, content: str, sha256: str) -> dict:
    return {
        "id": int(doc_id),
        "path": str(path.absolute()),
        "sha256": sha256,
        "content": content,
        "word_count": int(count_words(content)),
    }


def collect_paths(paths: Sequence[Path]) -> list[Path]:
    return sorted(
        {path.absolute() for path in paths},
        key=lambda path: str(path).casefold(),
    )


def path_in_folder(path: Path, folder: Path) -> bool:
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False


def assign_term_ids(documents: Sequence[dict]) -> dict[str, int]:
    """Pass 1 of rebuild: collect all distinct normalized tokens.

    Sorting makes IDs deterministic for a clean rebuild, which helps debugging
    and reproducible indexes.
    """

    vocabulary: set[str] = set()
    for document in documents:
        vocabulary.update(token for token, _start, _end in iter_tokens(document["content"]))

    return {term: term_id for term_id, term in enumerate(sorted(vocabulary))}


def build_postings_for_document(doc_id: int, content: str, term_to_id: dict[str, int]) -> list[dict]:
    spans_by_term: dict[int, list[int]] = {}

    for token, start, end in iter_tokens(content):
        term_id = term_to_id[token]
        spans_by_term.setdefault(term_id, []).extend((int(start), int(end - start)))

    rows: list[dict] = []
    for term_id, flat_positions in sorted(spans_by_term.items()):
        positions = np.asarray(flat_positions, dtype=np.int32)
        rows.append(
            {
                "doc_id": int(doc_id),
                "term_id": int(term_id),
                "positions": positions,
            }
        )
    return rows


def frequency_rows_from_postings(postings: Sequence[dict]) -> list[dict]:
    return [
        {
            "doc_id": int(row["doc_id"]),
            "term_id": int(row["term_id"]),
            "occurrence_count": int(len(row["positions"]) // 2),
        }
        for row in postings
    ]


def fts_document_rows(doc_id: int, content: str, term_to_id: dict[str, int]) -> list[dict]:
    """One FTS row per distinct ``(doc_id, normalized_token)`` pair."""

    terms = sorted(
        {token for token, _start, _end in iter_tokens(content)},
        key=lambda token: term_to_id[token],
    )
    return [
        {
            "doc_id": int(doc_id),
            "term_id": int(term_to_id[token]),
            FTS_TOKEN_COLUMN: token,
        }
        for token in terms
    ]


def refresh_fts_index(table: object) -> bool:
    """Create/rebuild the Tantivy FTS index using the raw single-token analyzer."""

    if not hasattr(table, "create_fts_index"):
        return False

    # Do not fall back to the default tokenizer: this index is explicitly a
    # single-token index and the raw analyzer preserves complete tokens.
    attempts = (
        {
            "replace": True,
            "base_tokenizer": "raw",
            "stem": False,
            "remove_stop_words": False,
            "ascii_folding": False,
        },
        {
            "replace": True,
            "base_tokenizer": "raw",
        },
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            table.create_fts_index(FTS_TOKEN_COLUMN, **kwargs)
            return True
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

    logger.error("Could not create raw-token FTS index: %r", last_error)
    return False


# -----------------------------------------------------------------------------
# FTS/Tantivy candidate filtering
# -----------------------------------------------------------------------------

def fts_escape_token(term: str) -> str:
    special = set(r'+-=&|><!(){}[]^\"~*?:/')
    return "".join("\\" + char if char in special else char for char in term)


def _has_regex_syntax_except_wildcards(term: str) -> bool:
    return any(char in "\\.^$+{}[]|()" for char in term)


def fts_query_for_keyword(query_term: str) -> str | None:
    query = query_term.strip().casefold()
    if not query:
        return None

    # Simple wildcard syntax can be passed to Tantivy. The authoritative
    # Python matcher remains responsible for final full-match semantics.
    if ("*" in query or "?" in query) and not _has_regex_syntax_except_wildcards(query):
        return query
    if not _has_regex_syntax_except_wildcards(query):
        return fts_escape_token(query)
    if "/" in query:
        return None
    return f"/{query}/"


def fts_match_query(query_term: str, fuzzy_distance: int) -> tuple[object, str] | None:
    query = query_term.strip().casefold()
    if not query:
        return None

    try:
        from lancedb.query import MatchQuery
    except Exception as exc:
        raise RuntimeError("LanceDB MatchQuery is unavailable") from exc

    attempts = (
        {
            "query": query,
            "column": FTS_TOKEN_COLUMN,
            "fuzziness": fuzzy_distance,
            "max_expansions": 50,
            "prefix_length": 0,
        },
        {
            "query": query,
            "column": FTS_TOKEN_COLUMN,
            "fuzziness": fuzzy_distance,
        },
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return MatchQuery(**kwargs), (
                f"MatchQuery(query={query!r}, fuzziness={fuzzy_distance}, "
                f"max_expansions={kwargs.get('max_expansions', 'default')}, "
                f"prefix_length={kwargs.get('prefix_length', 'default')})"
            )
        except TypeError as exc:
            last_error = exc
    raise RuntimeError(f"Could not construct LanceDB MatchQuery: {last_error}")


def all_document_ids(index: "SearchIndex") -> set[int]:
    rows = query_rows(index.docs, columns=["id"])
    return {int(row["id"]) for row in rows}


def fts_candidate_docs(
    index: "SearchIndex",
    query_terms: list[str],
    mode: SearchMode,
    fuzzy_distance: int,
    report: SearchReport,
) -> set[int] | None:
    """Return candidate document IDs from the one-token Tantivy index.

    Search results are immediately reduced to ``doc_id`` sets and intersected.
    The FTS phase is never authoritative; a disabled/failed FTS lookup falls
    back to all documents so the positional matcher remains correct.
    """

    if not index.fts_enabled:
        return None

    candidate_docs: set[int] | None = None
    all_docs_cache: set[int] | None = None
    cache: dict[tuple[str, SearchMode, int], set[int]] = {}

    def fallback_all_docs(status: str, entry: dict) -> set[int]:
        nonlocal all_docs_cache
        if all_docs_cache is None:
            all_docs_cache = all_document_ids(index)
        entry["status"] = status
        return set(all_docs_cache)

    for query_term in query_terms:
        normalized_query = query_term.strip().casefold()
        started = time.perf_counter()
        entry = {
            "query_term": query_term,
            "variants": None,
            "candidate_docs": None,
            "status": "used",
            "time_ms": 0.0,
            "query_text": None,
        }

        cache_key = (normalized_query, mode, int(fuzzy_distance))
        try:
            if cache_key in cache:
                docs_for_keyword = set(cache[cache_key])
                entry["status"] = "cache"
                entry["query_text"] = "cached"
            else:
                if mode == "fuzzy":
                    query_object, query_text = fts_match_query(
                        query_term, fuzzy_distance
                    )
                else:
                    query_text = fts_query_for_keyword(query_term)
                    query_object = None

                entry["query_text"] = query_text
                if query_object is not None:
                    setup_started = time.perf_counter()
                    search = index.fts_docs.search(query_object)
                    count_started = time.perf_counter()
                    fts_row_limit = int(index.fts_docs.count_rows())
                    _record_lancedb_scalar_query(
                        table=index.fts_docs,
                        operation="Tantivy candidate count_rows",
                        execution_ms=(time.perf_counter() - count_started) * 1000.0,
                    )
                    limit = max(fts_row_limit, 1)
                    setup_ms = (time.perf_counter() - setup_started) * 1000.0
                elif query_text is not None:
                    setup_started = time.perf_counter()
                    search = index.fts_docs.search(query_text, query_type="fts")
                    count_started = time.perf_counter()
                    fts_row_limit = int(index.fts_docs.count_rows())
                    _record_lancedb_scalar_query(
                        table=index.fts_docs,
                        operation="Tantivy candidate count_rows",
                        execution_ms=(time.perf_counter() - count_started) * 1000.0,
                    )
                    limit = max(fts_row_limit, 1)
                    setup_ms = (time.perf_counter() - setup_started) * 1000.0
                else:
                    docs_for_keyword = fallback_all_docs("fallback: broad", entry)
                    search = None
                    setup_ms = 0.0

                if search is not None:
                    started_exec = time.perf_counter()
                    arrow, used_score_fallback = _fts_to_arrow_without_score(
                        search, fts_row_limit
                    )
                    execution_ms = (time.perf_counter() - started_exec) * 1000.0
                    arrow_bytes = int(getattr(arrow, "nbytes", 0) or 0)
                    started_py = time.perf_counter()
                    rows = arrow.to_pylist()
                    to_pylist_ms = (time.perf_counter() - started_py) * 1000.0
                    table_key = _table_cache_key(index.fts_docs)
                    cache_state = "warm" if table_key in _DB_CACHE_SEEN_TABLES else "cold"
                    _DB_CACHE_SEEN_TABLES.add(table_key)
                    fragment_file_count = _query_fragment_file_count(search, arrow, index.fts_docs)
                    report = _ACTIVE_DB_QUERY_REPORT.get()
                    if report is not None:
                        if used_score_fallback:
                            report.diagnostics.append(
                                "LanceDB FTS scanner flag is not exposed by the installed high-level query API; "
                                "_score was projected only as a compatibility fallback and dropped immediately."
                            )
                        report.db_query_setup_ms += setup_ms
                        report.db_execution_ms += execution_ms
                        report.db_to_pylist_ms += to_pylist_ms
                        report.db_python_object_creation_ms += to_pylist_ms
                        report.db_queries += 1
                        report.db_rows_returned += len(rows)
                        report.db_bytes_returned += arrow_bytes
                        if cache_state == "cold":
                            report.db_cache_cold_queries += 1
                        else:
                            report.db_cache_warm_queries += 1
                        if fragment_file_count is not None:
                            report.db_fragment_files_touched = (
                                int(fragment_file_count)
                                if report.db_fragment_files_touched is None
                                else report.db_fragment_files_touched + int(fragment_file_count)
                            )
                            report.db_fragment_file_counts_known += 1
                        report.db_query_breakdown.append({
                            "operation": "Tantivy candidate lookup",
                            "table": table_key,
                            "cache_state": cache_state,
                            "setup_ms": setup_ms,
                            "execution_ms": execution_ms,
                            "to_pylist_ms": to_pylist_ms,
                            "python_object_creation_ms": to_pylist_ms,
                            "rows": len(rows),
                            "bytes": arrow_bytes,
                            "fragment_files_touched": fragment_file_count,
                        })
                    docs_for_keyword = {int(row["doc_id"]) for row in rows}
                    if not docs_for_keyword:
                        docs_for_keyword = fallback_all_docs(
                            "fallback: empty", entry
                        )

                cache[cache_key] = set(docs_for_keyword)
        except Exception as exc:
            docs_for_keyword = fallback_all_docs(
                f"fallback: {type(exc).__name__}", entry
            )
            error_type = type(exc).__name__
            error_message = str(exc).strip() or repr(exc)
            entry["error_type"] = error_type
            entry["error_message"] = error_message
            report.diagnostics.append(
                f"Tantivy candidate lookup failed for keyword {query_term!r}: "
                f"{error_type}: {error_message} | FTS query: {entry['query_text']!r}"
            )
            logger.error(
                "Tantivy candidate lookup failed for keyword=%r\n%s",
                query_term,
                traceback.format_exc(),
            )

        entry["candidate_docs"] = len(docs_for_keyword)
        entry["time_ms"] = (time.perf_counter() - started) * 1000.0
        report.tantivy_by_term.append(entry)
        report.tantivy_ms += entry["time_ms"]

        intersection_started = time.perf_counter()
        if candidate_docs is None:
            candidate_docs = set(docs_for_keyword)
        else:
            candidate_docs.intersection_update(docs_for_keyword)
        report.candidate_intersection_ms += (
            time.perf_counter() - intersection_started
        ) * 1000.0

        if not candidate_docs:
            report.candidate_docs_before_positions = 0
            return set()

    report.candidate_docs_before_positions = len(candidate_docs or set())
    return candidate_docs or set()


# -----------------------------------------------------------------------------
# Vocabulary / variant resolution
# -----------------------------------------------------------------------------

def load_vocabulary(table: object) -> tuple[dict[str, int], dict[int, str], dict[int, int]]:
    rows = query_rows(
        table,
        columns=["term_id", "term", "token_length"],
    )
    term_to_id = {str(row["term"]): int(row["term_id"]) for row in rows}
    id_to_term = {int(row["term_id"]): str(row["term"]) for row in rows}
    length_by_id = {
        int(row["term_id"]): int(row["token_length"]) for row in rows
    }
    return term_to_id, id_to_term, length_by_id


def compile_regex(term: str) -> re.Pattern[str] | None:
    try:
        return re.compile(term.casefold())
    except re.error:
        return None


def compile_wildcard(term: str) -> re.Pattern[str] | None:
    parts: list[str] = []
    for char in term.casefold():
        if char == "*":
            parts.append(r"[^\s]+")
        elif char == "?":
            parts.append(r"[^\s]")
        else:
            parts.append(re.escape(char))
    try:
        return re.compile("".join(parts))
    except re.error:
        return None


def resolve_regex_variants(
    index: "SearchIndex",
    query_term: str,
) -> tuple[dict[int, int], str]:
    query = query_term.strip()
    if not query:
        return {}, "empty query"

    normalized = query.casefold()
    if not _has_regex_syntax_except_wildcards(query) and "*" not in query and "?" not in query:
        term_id = index.term_to_id.get(normalized)
        if term_id is None:
            return {}, "exact vocabulary lookup"
        return {int(term_id): 0}, "exact vocabulary lookup"

    if ("*" in query or "?" in query) and not _has_regex_syntax_except_wildcards(query):
        pattern = compile_wildcard(query)
        detail = "wildcard vocabulary scan"
    else:
        pattern = compile_regex(query)
        detail = "regex vocabulary scan"

    if pattern is None:
        return {}, detail

    return {
        int(term_id): 0
        for term_id, term in index.id_to_term.items()
        if pattern.fullmatch(term)
    }, detail


def resolve_fuzzy_variants_from_candidate_terms(
    index: "SearchIndex",
    query_terms: list[str],
    candidate_docs: set[int],
    config: SearchConfig,
    report: SearchReport,
) -> list[dict[int, int]]:
    """Resolve fuzzy variants from the in-memory vocabulary, without postings caches.

    FTS has already reduced the corpus to candidate documents. The vocabulary is
    small enough to keep in RAM, so fuzzy matching can use it directly without
    constructing a document-term cache from the entire postings table.
    """

    del candidate_docs  # FTS candidates are applied later by frequency planning.
    variants_by_term: list[dict[int, int]] = []
    started_total = time.perf_counter()

    for query_term in query_terms:
        started = time.perf_counter()
        variants, mode_detail = resolve_variants(index, query_term, "fuzzy", config)
        variants_by_term.append(variants)
        report.variant_resolution.append(
            {
                "query_term": query_term,
                "variants": len(variants),
                "time_ms": (time.perf_counter() - started) * 1000.0,
                "mode_detail": mode_detail,
            }
        )

    report.variant_resolution_ms = (time.perf_counter() - started_total) * 1000.0
    report.variant_rapidfuzz_ms = report.variant_resolution_ms
    report.variant_candidate_term_ids = len(index.term_to_id)
    report.runtime_candidate_terms = len(index.term_to_id)
    return variants_by_term


def resolve_variants(
    index: "SearchIndex",
    query_term: str,
    mode: SearchMode,
    config: SearchConfig,
) -> tuple[dict[int, int], str]:
    if mode == "regex":
        return resolve_regex_variants(index, query_term)

    query = query_term.strip()
    if not query or "*" in query or "?" in query:
        return {}, "fuzzy mode rejects wildcard syntax"

    normalized = query.casefold()
    lower = max(0, len(normalized) - config.fuzzy_distance)
    upper = len(normalized) + config.fuzzy_distance
    vocabulary_ids = [
        int(term_id)
        for term_id, token_length in index.term_lengths.items()
        if lower <= int(token_length) <= upper
    ]
    vocabulary_ids.sort()
    candidates = [index.id_to_term[term_id] for term_id in vocabulary_ids]
    matches = process.extract(
        normalized,
        candidates,
        scorer=Levenshtein.distance,
        score_cutoff=config.fuzzy_distance,
        score_hint=config.fuzzy_distance,
        limit=None,
    )
    matches.sort(key=lambda item: (item[1], item[0]))
    return (
        {int(index.term_to_id[term]): int(distance) for term, distance, _ in matches},
        "in-memory vocabulary fuzzy scan",
    )


# -----------------------------------------------------------------------------
# Lightweight frequency planning
# -----------------------------------------------------------------------------

def fetch_term_frequencies(
    index: "SearchIndex",
    query_terms: list[str],
    variants_by_term: list[dict[int, int]],
    candidate_docs: set[int],
    report: SearchReport | None = None,
) -> tuple[dict[int, dict[str, int]], int]:
    """Fetch only the needed frequency rows for the current candidate documents."""

    if not candidate_docs:
        return {}, 0

    all_variants = sorted(
        {int(term_id) for variants in variants_by_term for term_id in variants}
    )
    if not all_variants:
        return {}, 0

    frequency_by_doc_term: dict[int, dict[str, int]] = {
        int(doc_id): {} for doc_id in candidate_docs
    }
    normalized_query_terms = [term.casefold() for term in query_terms]
    variant_to_query_terms: dict[int, list[str]] = {}
    for query_term, variants in zip(normalized_query_terms, variants_by_term):
        for term_id in variants:
            variant_to_query_terms.setdefault(int(term_id), []).append(query_term)

    term_chunk_size = 500
    rows_seen = 0
    candidate_doc_ids = sorted(int(doc_id) for doc_id in candidate_docs)
    for offset in range(0, len(all_variants), term_chunk_size):
        term_chunk = all_variants[offset : offset + term_chunk_size]
        predicate = where_in("doc_id", candidate_doc_ids) + " AND " + where_in("term_id", term_chunk)
        rows = query_rows(
            index.frequencies,
            predicate,
            ["doc_id", "term_id", "occurrence_count"],
            operation="Frequency planning query",
        )
        for row in rows:
            doc_id = int(row["doc_id"])
            term_id = int(row["term_id"])
            count = int(row["occurrence_count"])
            if report is not None:
                report.runtime_frequency_hits += 1
            rows_seen += 1
            for query_term in variant_to_query_terms.get(term_id, []):
                target = frequency_by_doc_term.setdefault(doc_id, {})
                target[query_term] = target.get(query_term, 0) + count

    if report is not None:
        report.runtime_frequency_lookups += len(candidate_doc_ids) * len(all_variants)
    return frequency_by_doc_term, rows_seen


def choose_anchors(
    query_terms: list[str],
    frequency_by_doc_term: dict[int, dict[str, int]],
) -> dict[int, dict]:
    """Keep all-term documents and choose their rarest query term as anchor."""

    normalized = [term.casefold() for term in query_terms]
    unique_terms = list(dict.fromkeys(normalized))
    anchors: dict[int, dict] = {}
    priority = {term: position for position, term in enumerate(unique_terms)}

    for doc_id, counts in frequency_by_doc_term.items():
        if any(counts.get(term, 0) <= 0 for term in unique_terms):
            continue
        count, term = min(
            ((counts[term], term) for term in unique_terms),
            key=lambda item: (item[0], priority[item[1]]),
        )
        anchors[doc_id] = {
            "query_term": term,
            "occurrences": count,
        }
    return anchors


# -----------------------------------------------------------------------------
# Compact positional postings
# -----------------------------------------------------------------------------

def positions_to_numpy(value: object) -> np.ndarray:
    """Decode persisted flat (start, length) pairs into an int32 NumPy array."""

    if value is None:
        return np.empty(0, dtype=np.int32)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) % np.dtype(np.int32).itemsize:
            raise ValueError("Corrupt positional posting: byte length is not int32-aligned")
        array = np.frombuffer(raw, dtype=np.int32)
        if array.size % 2:
            raise ValueError("Corrupt positional posting: odd number of int32 offsets")
        return np.array(array, dtype=np.int32, copy=True)

    array = np.asarray(value, dtype=np.int32).reshape(-1)
    if array.size % 2:
        raise ValueError("Corrupt positional posting: odd number of int32 offsets")
    return np.array(array, dtype=np.int32, copy=True)


def fetch_postings(
    index: "SearchIndex",
    term_ids: set[int],
    candidate_docs: set[int],
    report: SearchReport | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """Fetch only positional postings needed for the current candidate documents."""

    if not term_ids or not candidate_docs:
        return {}

    grouped: dict[int, dict[int, np.ndarray]] = {}
    candidate_doc_ids = sorted(int(doc_id) for doc_id in candidate_docs)
    sorted_term_ids = sorted(int(value) for value in term_ids)
    term_chunk_size = 500

    for offset in range(0, len(sorted_term_ids), term_chunk_size):
        term_chunk = sorted_term_ids[offset : offset + term_chunk_size]
        predicate = where_in("doc_id", candidate_doc_ids) + " AND " + where_in("term_id", term_chunk)
        rows = query_rows(
            index.postings,
            predicate,
            ["doc_id", "term_id", "positions"],
            operation="Positional postings query",
        )
        for row in rows:
            term_id = int(row["term_id"])
            doc_id = int(row["doc_id"])
            grouped.setdefault(term_id, {})[doc_id] = positions_to_numpy(row["positions"])

        if report is not None:
            report.runtime_posting_lookups += len(candidate_doc_ids) * len(term_chunk)
            report.runtime_posting_hits += len(rows)

    for term_id in sorted_term_ids:
        grouped.setdefault(term_id, {})
    return grouped


def build_match_arrays(
    variants: dict[int, int],
    postings: dict[int, dict[int, np.ndarray]],
) -> dict[int, MatchArrays]:
    """Merge all accepted indexed variants into one NumPy representation per doc."""

    chunks: dict[int, list[tuple[np.ndarray, np.ndarray, int, int]]] = {}

    for term_id, edit_distance in variants.items():
        for doc_id, positions in postings.get(term_id, {}).items():
            if positions.size == 0:
                continue
            starts = positions[0::2]
            lengths = positions[1::2]
            ends = starts + lengths
            chunks.setdefault(doc_id, []).append(
                (
                    np.asarray(starts, dtype=np.int32),
                    np.asarray(ends, dtype=np.int32),
                    int(term_id),
                    int(edit_distance),
                )
            )

    result: dict[int, MatchArrays] = {}
    for doc_id, pieces in chunks.items():
        if not pieces:
            result[doc_id] = MatchArrays.empty()
            continue

        starts = np.concatenate([piece[0] for piece in pieces]).astype(np.int32, copy=False)
        ends = np.concatenate([piece[1] for piece in pieces]).astype(np.int32, copy=False)
        term_ids = np.concatenate(
            [np.full(piece[0].size, piece[2], dtype=np.int32) for piece in pieces]
        )
        edit_distances = np.concatenate(
            [
                np.full(piece[0].size, piece[3], dtype=np.int32)
                for piece in pieces
            ]
        )

        # One physical occurrence belongs to one indexed term. In fuzzy mode,
        # multiple accepted query variants can converge on the same coordinates.
        # Keep the closest accepted variant at each coordinate.
        order = np.lexsort((edit_distances, term_ids, ends, starts))
        starts = starts[order]
        ends = ends[order]
        term_ids = term_ids[order]
        edit_distances = edit_distances[order]

        keep = np.ones(starts.size, dtype=bool)
        if starts.size > 1:
            same_position = (starts[1:] == starts[:-1]) & (ends[1:] == ends[:-1])
            keep[1:] = ~same_position

        result[doc_id] = MatchArrays(
            starts=starts[keep],
            ends=ends[keep],
            term_ids=term_ids[keep],
            edit_distances=edit_distances[keep],
        )

    return result


def build_matches_for_document(
    query_term: str,
    matches: MatchArrays,
    id_to_term: dict[int, str],
    mode: SearchMode,
) -> list[dict]:
    if len(matches) == 0:
        return []

    query_len = len(query_term.strip().casefold())
    output: list[dict] = []
    for start, end, term_id, edit_distance in zip(
        matches.starts.tolist(),
        matches.ends.tolist(),
        matches.term_ids.tolist(),
        matches.edit_distances.tolist(),
    ):
        indexed_term = id_to_term[int(term_id)]
        max_len = max(query_len, len(indexed_term))
        score = (
            100.0
            if mode == "regex" or max_len == 0
            else 100.0 * (1.0 - int(edit_distance) / max_len)
        )
        output.append(
            {
                "start": int(start),
                "end": int(end),
                "score": float(score),
                "edit_distance": int(edit_distance),
                "query_term": query_term,
                "indexed_term": indexed_term,
                "term_id": int(term_id),
            }
        )
    return output


# -----------------------------------------------------------------------------
# NumPy proximity engine
# -----------------------------------------------------------------------------

def distance_between(first: dict, second: dict) -> int:
    if first["end"] <= second["start"]:
        return second["start"] - first["end"]
    if second["end"] <= first["start"]:
        return first["start"] - second["end"]
    return 0


def _empty_chain_result() -> list[list[dict]]:
    return []


def find_proximity_chains_numpy(
    matches_by_term: list[MatchArrays],
    query_terms: list[str],
    id_to_term: dict[int, str],
    config: SearchConfig,
    anchor_term: str | None = None,
) -> list[list[dict]]:
    """Find proximity chains using flat NumPy arrays.

    Every query-term occurrence is represented by numeric arrays.  Connected
    components are detected from vectorized gap arithmetic; the final chain is
    materialized back to dictionaries only for the public ``SearchHit`` output.
    """

    if not matches_by_term or any(len(matches) == 0 for matches in matches_by_term):
        return _empty_chain_result()

    # Required multiplicities are based on normalized user keywords.  Duplicate
    # GUI slots therefore require separate physical occurrences.
    required_counts: dict[str, int] = {}
    for query_term in query_terms:
        normalized = query_term.casefold()
        required_counts[normalized] = required_counts.get(normalized, 0) + 1

    unique_terms = list(required_counts)
    term_code = {term: code for code, term in enumerate(unique_terms)}
    display_terms: dict[str, str] = {}
    for query_term in query_terms:
        display_terms.setdefault(query_term.casefold(), query_term)

    starts_parts: list[np.ndarray] = []
    ends_parts: list[np.ndarray] = []
    term_code_parts: list[np.ndarray] = []
    indexed_term_parts: list[np.ndarray] = []
    edit_parts: list[np.ndarray] = []

    for query_term, matches in zip(query_terms, matches_by_term):
        code = term_code[query_term.casefold()]
        if len(matches) == 0:
            continue
        starts_parts.append(matches.starts)
        ends_parts.append(matches.ends)
        term_code_parts.append(
            np.full(len(matches), code, dtype=np.int32)
        )
        indexed_term_parts.append(matches.term_ids)
        edit_parts.append(matches.edit_distances)

    if not starts_parts:
        return _empty_chain_result()

    starts = np.concatenate(starts_parts).astype(np.int32, copy=False)
    ends = np.concatenate(ends_parts).astype(np.int32, copy=False)
    codes = np.concatenate(term_code_parts).astype(np.int16, copy=False)
    indexed_term_ids = np.concatenate(indexed_term_parts).astype(np.int32, copy=False)
    edit_distances = np.concatenate(edit_parts).astype(np.int32, copy=False)

    # Sort by position, then normalized query code. Duplicate query slots for
    # the same normalized keyword are intentionally adjacent and deduplicated.
    order = np.lexsort((edit_distances, indexed_term_ids, codes, ends, starts))
    starts = starts[order]
    ends = ends[order]
    codes = codes[order]
    indexed_term_ids = indexed_term_ids[order]
    edit_distances = edit_distances[order]

    keep = np.ones(starts.size, dtype=bool)
    if starts.size > 1:
        duplicate_event = (
            (starts[1:] == starts[:-1])
            & (ends[1:] == ends[:-1])
            & (codes[1:] == codes[:-1])
        )
        keep[1:] = ~duplicate_event
    starts = starts[keep]
    ends = ends[keep]
    codes = codes[keep]
    indexed_term_ids = indexed_term_ids[keep]
    edit_distances = edit_distances[keep]

    if starts.size == 0:
        return _empty_chain_result()

    # Because events are sorted by start, the gap to the next event is exactly
    # ``max(0, next_start - current_end)``.  This is vectorized for the entire
    # document instead of repeatedly calling a Python dict-based distance helper.
    if starts.size > 1:
        gaps = np.maximum(
            np.zeros(starts.size - 1, dtype=np.int32),
            starts[1:] - ends[:-1],
        )
        split_points = np.flatnonzero(gaps > config.max_distance_chars) + 1
    else:
        split_points = np.empty(0, dtype=np.int64)

    component_starts = np.concatenate(
        [np.array([0], dtype=np.int64), split_points]
    )
    component_ends = np.concatenate(
        [
            split_points.astype(np.int64, copy=False) - 1,
            np.array([starts.size - 1], dtype=np.int64),
        ]
    )

    anchor_norm = anchor_term.casefold() if anchor_term else None
    if anchor_norm not in required_counts:
        # Fallback if the planner did not choose an anchor.
        counts = np.bincount(
            codes.astype(np.int64),
            minlength=len(unique_terms),
        )
        anchor_code = min(
            range(len(unique_terms)),
            key=lambda code: (int(counts[code]), unique_terms[code]),
        )
    else:
        anchor_code = term_code[anchor_norm]

    chains: list[list[dict]] = []
    limit = config.max_proximity_chains_per_document

    for component_start, component_end in zip(
        component_starts.tolist(), component_ends.tolist()
    ):
        component_codes = codes[component_start : component_end + 1]
        anchor_indexes = np.flatnonzero(component_codes == anchor_code)
        if anchor_indexes.size == 0:
            continue

        counts = np.zeros(len(unique_terms), dtype=np.int32)
        last_complete_index = -1
        for index, code in enumerate(component_codes.tolist()):
            counts[code] += 1
            if all(
                counts[code_index] >= required
                for code_index, required in enumerate(
                    required_counts.values()
                )
            ):
                last_complete_index = index
                counts.fill(0)

        if last_complete_index < 0:
            continue
        if int(anchor_indexes[0]) > last_complete_index:
            continue

        absolute_end = component_start + last_complete_index
        chain_starts = starts[component_start : absolute_end + 1]
        chain_ends = ends[component_start : absolute_end + 1]
        chain_codes = codes[component_start : absolute_end + 1]
        chain_term_ids = indexed_term_ids[component_start : absolute_end + 1]
        chain_edits = edit_distances[component_start : absolute_end + 1]

        chain: list[dict] = []
        for start, end, code, term_id, edit_distance in zip(
            chain_starts.tolist(),
            chain_ends.tolist(),
            chain_codes.tolist(),
            chain_term_ids.tolist(),
            chain_edits.tolist(),
        ):
            query_term = display_terms[unique_terms[int(code)]]
            indexed_term = id_to_term[int(term_id)]
            max_len = max(len(query_term.strip().casefold()), len(indexed_term))
            score = (
                100.0
                if max_len == 0
                else 100.0 * (1.0 - int(edit_distance) / max_len)
            )
            chain.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "score": float(score),
                    "edit_distance": int(edit_distance),
                    "query_term": query_term,
                    "indexed_term": indexed_term,
                    "term_id": int(term_id),
                }
            )

        if chain:
            chains.append(chain)
        if limit is not None and len(chains) >= limit:
            break

    return chains[:limit] if limit is not None else chains


# -----------------------------------------------------------------------------
# Snippets and result documents
# -----------------------------------------------------------------------------

MATCH_MARKER_OPEN = "\ue000"
MATCH_MARKER_CLOSE = "\ue001"


def build_snippet(content: str, chain: list[dict], context_chars: int) -> str:
    """Build a snippet with private match markers for GUI-only highlighting.

    The GUI removes these non-printing markers before displaying/copying the
    result, so users see only the original snippet text.
    """
    start = min(m["start"] for m in chain)
    end = max(m["end"] for m in chain)
    snippet_start = max(0, start - context_chars)
    snippet_end = min(len(content), end + context_chars)
    snippet = (
        content[snippet_start:snippet_end]
        .replace("\r", " ")
        .replace("\n", " ")
    )

    for a, b in reversed(sorted({(m["start"], m["end"]) for m in chain})): 
        if a < snippet_start or b > snippet_end:
            continue
        left = a - snippet_start
        right = b - snippet_start
        snippet = (
            snippet[:left]
            + MATCH_MARKER_OPEN
            + snippet[left:right]
            + MATCH_MARKER_CLOSE
            + snippet[right:]
        )

    prefix = "..." if snippet_start else ""
    suffix = "..." if snippet_end < len(content) else ""
    return prefix + snippet + suffix


def fetch_documents(
    index: "SearchIndex",
    doc_ids: Sequence[int],
    report: SearchReport | None = None,
) -> dict[int, dict]:
    """Load final-hit source text from disk, falling back to LanceDB if needed.

    The index stores character offsets into the indexed UTF-8 text. To preserve
    exact snippet semantics, the source file is verified against the indexed
    SHA-256 before its text is used. If the source is unavailable or has changed
    without a synchronization pass, the persisted document row remains the
    correctness fallback.
    """

    if not doc_ids:
        return {}

    documents: dict[int, dict] = {}
    fallback_ids: list[int] = []

    for raw_doc_id in doc_ids:
        doc_id = int(raw_doc_id)
        metadata = index.document_metadata.get(doc_id)
        if metadata is None:
            fallback_ids.append(doc_id)
            continue

        path = Path(str(metadata["path"]))
        try:
            content, sha256 = read_text_with_sha256(path)
            if str(sha256) != str(metadata.get("sha256", "")):
                raise ValueError("source hash differs from indexed hash")
            document = dict(metadata)
            document["content"] = content
            documents[doc_id] = document
            if report is not None:
                report.runtime_source_file_reads += 1
        except Exception as exc:
            logger.warning(
                "Using LanceDB document fallback for doc_id=%s path=%s: %s",
                doc_id,
                path,
                exc,
            )
            fallback_ids.append(doc_id)
            if report is not None:
                report.runtime_lancedb_document_fallbacks += 1

    if fallback_ids:
        rows = query_rows_in(
            index.docs,
            "id",
            fallback_ids,
            ["id", "path", "sha256", "content"],
            operation="Result document fallback",
        )
        documents.update({int(row["id"]): row for row in rows})

    return documents


# -----------------------------------------------------------------------------
# Rebuild / incremental synchronization
# -----------------------------------------------------------------------------

def create_empty_index(db: object):
    for name in (
        DOCS_TABLE,
        POSTINGS_TABLE,
        *LEGACY_POSTINGS_TABLES,
        VOCABULARY_TABLE,
        FREQUENCIES_TABLE,
        FTS_TABLE,
    ):
        if table_exists(db, name):
            db.drop_table(name)

    return (
        db.create_table(DOCS_TABLE, schema=DOCS_SCHEMA),
        db.create_table(POSTINGS_TABLE, schema=POSTINGS_SCHEMA),
        db.create_table(VOCABULARY_TABLE, schema=VOCABULARY_SCHEMA),
        db.create_table(FREQUENCIES_TABLE, schema=FREQUENCIES_SCHEMA),
        db.create_table(FTS_TABLE, schema=FTS_SCHEMA),
    )


def rebuild_index(
    db: object,
    paths: list[Path],
    *,
    progress_callback: ProgressCallback | None = None,
):
    """Clean rebuild in two passes.

    Pass 1 stores documents and creates the deterministic token -> term_id map.
    Pass 2 creates postings/frequencies/FTS rows using those numeric IDs.
    """

    docs, postings_table, vocabulary_table, frequencies_table, fts_table = create_empty_index(db)

    # Pass 1: read documents and assign vocabulary IDs.
    source_paths = collect_paths(paths)
    _notify_progress(progress_callback, "Reading text files", 0.0)
    documents: list[dict] = []
    doc_id = 0
    total_paths = max(len(source_paths), 1)
    for position, path in enumerate(source_paths, start=1):
        try:
            content, sha256 = read_text_with_sha256(path)
        except Exception:
            logger.exception("Could not read %s during rebuild", path)
            continue
        documents.append(document_row(doc_id, path, content, sha256))
        doc_id += 1
        _notify_progress(
            progress_callback,
            "Reading text files",
            0.35 * position / total_paths,
        )

    _notify_progress(progress_callback, "Building vocabulary and index", 0.40)
    term_to_id = assign_term_ids(documents)
    _notify_progress(progress_callback, "Building vocabulary and index", 0.50)

    # Persist vocabulary.
    vocabulary_rows = [
        {
            "term_id": int(term_id),
            "term": term,
            "token_length": int(len(term)),
        }
        for term, term_id in sorted(term_to_id.items(), key=lambda item: item[1])
    ]
    add_rows(vocabulary_table, vocabulary_rows, VOCABULARY_SCHEMA)
    term_lengths = {
        int(term_id): int(len(term))
        for term, term_id in term_to_id.items()
    }

    # Pass 2: build numeric postings/frequencies/FTS rows.
    postings: list[dict] = []
    frequencies: list[dict] = []
    fts_rows: list[dict] = []
    doc_word_counts: dict[int, int] = {}

    for document in documents:
        doc_id = int(document["id"])
        document_postings = build_postings_for_document(
            doc_id,
            document["content"],
            term_to_id,
        )
        postings.extend(document_postings)
        frequencies.extend(frequency_rows_from_postings(document_postings))
        doc_word_counts[doc_id] = int(
            sum(len(row["positions"]) // 2 for row in document_postings)
        )
        fts_rows.extend(
            fts_document_rows(doc_id, document["content"], term_to_id)
        )

    _notify_progress(progress_callback, "Writing index tables", 0.55)
    add_rows(docs, documents, DOCS_SCHEMA)
    add_rows(postings_table, postings, POSTINGS_SCHEMA)
    add_rows(frequencies_table, frequencies, FREQUENCIES_SCHEMA)
    _notify_progress(progress_callback, "Writing index tables", 0.78)
    add_rows(fts_table, fts_rows, FTS_SCHEMA)
    _notify_progress(progress_callback, "Creating FTS index", 0.88)
    fts_enabled = refresh_fts_index(fts_table)
    _notify_progress(progress_callback, "Indexing complete", 1.0)

    return (
        docs,
        postings_table,
        vocabulary_table,
        frequencies_table,
        fts_table,
        term_to_id,
        fts_enabled,
        doc_word_counts,
    )


def remove_unused_terms(index: "SearchIndex", affected_term_ids: set[int]) -> set[int]:
    """Delete vocabulary rows for terms with no remaining postings."""

    if not affected_term_ids:
        return set()

    surviving: set[int] = set()
    sorted_terms = sorted(affected_term_ids)
    for offset in range(0, len(sorted_terms), 500):
        chunk = sorted_terms[offset : offset + 500]
        rows = query_rows_in(
            index.postings,
            "term_id",
            chunk,
            ["term_id"],
        )
        surviving.update(int(row["term_id"]) for row in rows)

    obsolete = affected_term_ids - surviving
    if obsolete:
        delete_rows_in(index.vocabulary, "term_id", sorted(obsolete))
        for term_id in obsolete:
            term = index.id_to_term.pop(term_id, None)
            if term is not None:
                index.term_to_id.pop(term, None)
            index.term_lengths.pop(term_id, None)
    return obsolete


def sync_index(
    index: "SearchIndex",
    paths: list[Path],
    *,
    remove_paths: set[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Incrementally synchronize documents and all derived index structures."""

    remove_paths = {str(Path(p).absolute()) for p in (remove_paths or set())}
    source_paths = collect_paths(paths)
    current = {str(path.absolute()): path for path in source_paths}
    _notify_progress(progress_callback, "Reading text files", 0.0)
    old_rows = query_rows(index.docs, columns=["id", "path", "sha256"])
    old_by_path = {str(Path(row["path"]).absolute()): row for row in old_rows}
    next_id = max((int(row["id"]) for row in old_rows), default=-1) + 1

    removed_ids = {
        int(old_by_path[path]["id"])
        for path in remove_paths
        if path in old_by_path
    }

    changed_docs: list[dict] = []
    added_docs: list[dict] = []
    affected_ids = set(removed_ids)
    stats = {
        "added": 0,
        "changed": 0,
        "removed": len(removed_ids),
        "unchanged": 0,
    }

    current_items = list(current.items())
    total_current = max(len(current_items), 1)
    for position, (path_str, path) in enumerate(current_items, start=1):
        try:
            content, sha256 = read_text_with_sha256(path)
        except Exception:
            logger.exception("Could not read %s during sync", path)
            continue

        old = old_by_path.get(path_str)
        if old is None:
            added_docs.append(document_row(next_id, path, content, sha256))
            next_id += 1
            stats["added"] += 1
        elif old["sha256"] == sha256:
            stats["unchanged"] += 1
        else:
            doc_id = int(old["id"])
            changed_docs.append(document_row(doc_id, path, content, sha256))
            affected_ids.add(doc_id)
            stats["changed"] += 1

        _notify_progress(
            progress_callback,
            "Reading text files",
            0.30 * position / total_current,
        )

    _notify_progress(progress_callback, "Building vocabulary and index", 0.35)
    # Identify old terms before deleting old postings.
    affected_term_ids: set[int] = set()
    if affected_ids:
        old_postings = query_rows_in(
            index.postings,
            "doc_id",
            sorted(affected_ids),
            ["term_id"],
        )
        affected_term_ids.update(int(row["term_id"]) for row in old_postings)
        delete_rows_in(index.postings, "doc_id", sorted(affected_ids))
        delete_rows_in(index.frequencies, "doc_id", sorted(affected_ids))

    for doc_id in removed_ids:
        index.doc_word_counts.pop(doc_id, None)
    for document in changed_docs:
        index.doc_word_counts.pop(int(document["id"]), None)

    if removed_ids:
        delete_rows_in(index.docs, "id", sorted(removed_ids))
        delete_rows_in(index.fts_docs, "doc_id", sorted(removed_ids))

    if changed_docs:
        changed_ids = sorted(int(row["id"]) for row in changed_docs)
        delete_rows_in(index.docs, "id", changed_ids)
        delete_rows_in(index.fts_docs, "doc_id", changed_ids)
        add_rows(index.docs, changed_docs, DOCS_SCHEMA)

    if added_docs:
        add_rows(index.docs, added_docs, DOCS_SCHEMA)

    # Assign IDs to any new vocabulary terms before constructing new postings.
    next_term_id = max(index.term_to_id.values(), default=-1) + 1
    new_terms: list[tuple[str, int]] = []
    pending_terms: set[str] = set()
    for document in changed_docs + added_docs:
        pending_terms.update(token for token, _start, _end in iter_tokens(document["content"]))

    for token in sorted(pending_terms):
        if token not in index.term_to_id:
            term_id = int(next_term_id)
            next_term_id += 1
            index.term_to_id[token] = term_id
            index.id_to_term[term_id] = token
            index.term_lengths[term_id] = len(token)
            new_terms.append((token, term_id))

    if new_terms:
        add_rows(
            index.vocabulary,
            [
                {
                    "term_id": int(term_id),
                    "term": token,
                    "token_length": int(len(token)),
                }
                for token, term_id in new_terms
            ],
            VOCABULARY_SCHEMA,
        )

    _notify_progress(progress_callback, "Building vocabulary and index", 0.48)

    new_postings: list[dict] = []
    new_frequencies: list[dict] = []
    new_fts_rows: list[dict] = []
    for document in changed_docs + added_docs:
        doc_id = int(document["id"])
        document_postings = build_postings_for_document(
            doc_id,
            document["content"],
            index.term_to_id,
        )
        new_postings.extend(document_postings)
        new_frequencies.extend(
            frequency_rows_from_postings(document_postings)
        )
        index.doc_word_counts[doc_id] = int(
            sum(len(row["positions"]) // 2 for row in document_postings)
        )
        new_fts_rows.extend(
            fts_document_rows(doc_id, document["content"], index.term_to_id)
        )
        affected_term_ids.update(int(row["term_id"]) for row in document_postings)

    _notify_progress(progress_callback, "Writing index tables", 0.68)
    add_rows(index.postings, new_postings, POSTINGS_SCHEMA)
    add_rows(index.frequencies, new_frequencies, FREQUENCIES_SCHEMA)
    _notify_progress(progress_callback, "Writing index tables", 0.80)

    if changed_docs:
        # Old rows for changed documents were removed above.
        add_rows(index.fts_docs, new_fts_rows, FTS_SCHEMA)
    elif added_docs:
        add_rows(index.fts_docs, new_fts_rows, FTS_SCHEMA)

    remove_unused_terms(index, affected_term_ids)

    if changed_docs or added_docs or removed_ids:
        _notify_progress(progress_callback, "Creating FTS index", 0.90)
        index.fts_enabled = refresh_fts_index(index.fts_docs)
        index.document_metadata, index.doc_word_counts = build_runtime_search_caches(index.docs)

    _notify_progress(progress_callback, "Indexing complete", 1.0)
    return stats


# -----------------------------------------------------------------------------
# Search pipeline
# -----------------------------------------------------------------------------

def search_index(
    index: "SearchIndex",
    terms: list[str],
    mode: SearchMode,
    config: SearchConfig,
    max_results: int | None = MAX_RESULTS,
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[SearchHit], SearchReport]:
    started_total = time.perf_counter()
    report = SearchReport(mode=mode)
    db_report_token = _ACTIVE_DB_QUERY_REPORT.set(report)
    try:
        query_terms = [term.strip() for term in terms if term.strip()]
        _notify_progress(progress_callback, "Candidate lookup", 0.02)
        if not query_terms:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            return [], report

        # 1. Tantivy candidate filter.
        candidate_docs = fts_candidate_docs(
            index,
            query_terms,
            mode,
            config.fuzzy_distance,
            report,
        )
        if candidate_docs is None:
            candidate_docs = all_document_ids(index)
            report.candidate_docs_before_positions = len(candidate_docs)
            report.diagnostics.append(
                f"Tantivy unavailable; using all {len(candidate_docs):,} document(s) as candidates."
            )
        if not candidate_docs:
            _notify_progress(progress_callback, "Candidate lookup complete", 1.0)
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            return [], report

        _notify_progress(progress_callback, "Variant resolution", 0.20)

        # 2. Resolve user keywords to numeric vocabulary variants.
        variant_resolution_started = time.perf_counter()
        if mode == "fuzzy":
            variants_by_term = resolve_fuzzy_variants_from_candidate_terms(
                index,
                query_terms,
                candidate_docs,
                config,
                report,
            )
        else:
            variants_by_term = []
            for query_term in query_terms:
                started = time.perf_counter()
                variants, mode_detail = resolve_variants(
                    index, query_term, mode, config
                )
                variants_by_term.append(variants)
                report.variant_resolution.append(
                    {
                        "query_term": query_term,
                        "variants": len(variants),
                        "time_ms": (time.perf_counter() - started) * 1000.0,
                        "mode_detail": mode_detail,
                    }
                )
        if mode != "fuzzy":
            report.variant_resolution_ms = (
                time.perf_counter() - variant_resolution_started
            ) * 1000.0

        if any(not variants for variants in variants_by_term):
            _notify_progress(progress_callback, "Variant resolution complete", 0.30)
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            return [], report

        _notify_progress(progress_callback, "Frequency planning", 0.32)

        # 3. Cheap count-based document planning.
        frequency_started = time.perf_counter()
        frequency_by_doc_term, frequency_rows = fetch_term_frequencies(
            index,
            query_terms,
            variants_by_term,
            candidate_docs,
            report,
        )
        report.frequency_ms = (time.perf_counter() - frequency_started) * 1000.0
        report.frequency_rows = frequency_rows
        report.frequency_docs = len(frequency_by_doc_term)
        report.frequency_candidate_docs = len(candidate_docs)
        _notify_progress(progress_callback, "Positional postings", 0.48)

        anchor_started = time.perf_counter()
        anchors = choose_anchors(query_terms, frequency_by_doc_term)
        report.anchor_selection_ms = (time.perf_counter() - anchor_started) * 1000.0
        report.anchor_by_doc = [
            {"doc_id": doc_id, **anchor}
            for doc_id, anchor in sorted(anchors.items())
        ]

        candidate_docs = set(anchors)
        report.frequency_candidate_docs = len(candidate_docs)
        report.candidate_docs_before_positions = len(candidate_docs)
        if not candidate_docs:
            _notify_progress(progress_callback, "Frequency planning complete", 1.0)
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            return [], report

        # 4. Fetch positions for only the candidate docs.
        required_term_ids = {
            int(term_id)
            for variants in variants_by_term
            for term_id in variants
        }
        postings_started = time.perf_counter()
        postings = fetch_postings(index, required_term_ids, candidate_docs, report)
        report.postings_ms = (time.perf_counter() - postings_started) * 1000.0
        report.postings_terms = len(required_term_ids)
        report.postings_rows = sum(len(doc_map) for doc_map in postings.values())
        _notify_progress(progress_callback, "Match building", 0.62)

        # 5. Build compact NumPy matches per query term.
        match_build_started = time.perf_counter()
        match_arrays_by_term: list[dict[int, MatchArrays]] = []
        for query_term, variants in zip(query_terms, variants_by_term):
            started = time.perf_counter()
            mapping = build_match_arrays(variants, postings)
            match_arrays_by_term.append(mapping)
            report.match_build_by_term.append(
                {
                    "query_term": query_term,
                    "matching_docs": len(mapping),
                    "occurrences": sum(len(items) for items in mapping.values()),
                    "time_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
        report.match_build_ms = (time.perf_counter() - match_build_started) * 1000.0

        # Authoritative document intersection happens on the compact mappings,
        # before the per-document proximity pass.
        intersection_started = time.perf_counter()
        for mapping in match_arrays_by_term:
            candidate_docs.intersection_update(mapping)
            if not candidate_docs:
                break
        report.authoritative_intersection_ms = (
            time.perf_counter() - intersection_started
        ) * 1000.0
        report.candidate_docs_after_postings = len(candidate_docs)
        _notify_progress(progress_callback, "Proximity search", 0.74)
        if not candidate_docs:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            return [], report

        # 6. NumPy proximity engine.  Only the small remaining document set is
        # iterated individually; position calculations themselves operate on arrays.
        hit_ids: list[int] = []
        hit_chains: dict[int, list[list[dict]]] = {}
        proximity_started = time.perf_counter()

        for doc_id in sorted(candidate_docs):
            per_term = [mapping.get(doc_id, MatchArrays.empty()) for mapping in match_arrays_by_term]
            doc_started = time.perf_counter()
            anchor_info = anchors.get(doc_id)
            anchor_term = anchor_info["query_term"] if anchor_info else None
            chains = find_proximity_chains_numpy(
                per_term,
                query_terms,
                index.id_to_term,
                config,
                anchor_term=anchor_term,
            )
            doc_elapsed_ms = (time.perf_counter() - doc_started) * 1000.0
            report.proximity_docs_checked += 1
            report.proximity_max_doc_ms = max(report.proximity_max_doc_ms, doc_elapsed_ms)

            if chains:
                report.proximity_docs_with_hits += 1
                hit_ids.append(doc_id)
                hit_chains[doc_id] = chains
                if max_results is not None and len(hit_ids) >= max_results:
                    break

        report.proximity_ms = (time.perf_counter() - proximity_started) * 1000.0
        _notify_progress(progress_callback, "Fetching result documents", 0.88)

        # 7. Fetch full source text only for final hits.
        documents_started = time.perf_counter()
        documents = fetch_documents(index, hit_ids, report)
        report.documents_ms = (time.perf_counter() - documents_started) * 1000.0

        # 8. Materialize the compatibility API at the very end.
        result_started = time.perf_counter()
        hits: list[SearchHit] = []
        for doc_id in hit_ids:
            public_matches = [
                build_matches_for_document(
                    query_term,
                    mapping.get(doc_id, MatchArrays.empty()),
                    index.id_to_term,
                    mode,
                )
                for query_term, mapping in zip(query_terms, match_arrays_by_term)
            ]
            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    document=documents.get(doc_id, {}),
                    matches_by_term=public_matches,
                    chains=hit_chains[doc_id],
                    mode=mode,
                )
            )

        report.result_build_ms = (time.perf_counter() - result_started) * 1000.0
        report.final_documents = len(hits)
        report.total_ms = (time.perf_counter() - started_total) * 1000.0
        _notify_progress(progress_callback, "Search complete", 1.0)
        return hits, report
    finally:
        # Reset exactly once. ContextVar tokens are single-use; resetting the
        # same token both on the success path and in finally raises:
        # "<token ...> has already been used once".
        _ACTIVE_DB_QUERY_REPORT.reset(db_report_token)


# -----------------------------------------------------------------------------
# Document deletion
# -----------------------------------------------------------------------------

def _unique_deleted_destination(deleted_folder: Path, source_path: Path) -> Path:
    """Return a collision-free destination preserving the original filename."""

    destination = deleted_folder / source_path.name
    if not destination.exists():
        return destination

    suffix = source_path.suffix
    stem = source_path.stem
    counter = 1
    while True:
        candidate = deleted_folder / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def delete_document(
    index: "SearchIndex",
    doc_id: int,
    deleted_folder: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, object]:
    """Remove one document from every index table and move its source file."""

    _notify_progress(progress_callback, "Reading document", 0.02)
    rows = query_rows(index.docs, f"id = {int(doc_id)}", ["id", "path", "sha256"])
    if not rows:
        raise ValueError(f"Document id {doc_id} is not indexed.")

    document = rows[0]
    source_path = Path(str(document["path"])).absolute()
    deleted_folder = deleted_folder.absolute()
    deleted_folder.mkdir(parents=True, exist_ok=True)

    moved_to: Path | None = None
    _notify_progress(progress_callback, "Moving file to deleted", 0.15)
    if source_path.exists():
        # A file that is already in the destination folder does not need to be
        # moved again. This also makes deletion idempotent for unusual cases.
        if source_path.parent == deleted_folder:
            moved_to = source_path
        else:
            moved_to = _unique_deleted_destination(deleted_folder, source_path)
            try:
                shutil.move(str(source_path), str(moved_to))
            except Exception:
                logger.exception("Could not move %s to %s", source_path, moved_to)
                raise

    _notify_progress(progress_callback, "Preparing database removal", 0.30)
    deleted_word_count = int(index.doc_word_counts.get(int(doc_id), 0))
    affected_term_ids = {
        int(row["term_id"])
        for row in query_rows(
            index.postings,
            f"doc_id = {int(doc_id)}",
            ["term_id"],
        )
    }

    try:
        _notify_progress(progress_callback, "Removing document index", 0.45)
        delete_rows_in(index.postings, "doc_id", [int(doc_id)])
        delete_rows_in(index.frequencies, "doc_id", [int(doc_id)])
        delete_rows_in(index.fts_docs, "doc_id", [int(doc_id)])
        delete_rows_in(index.docs, "id", [int(doc_id)])

        _notify_progress(progress_callback, "Cleaning unused terms", 0.60)
        remove_unused_terms(index, affected_term_ids)
        index.doc_word_counts.pop(int(doc_id), None)

        _notify_progress(progress_callback, "Refreshing search index", 0.78)
        index.fts_enabled = refresh_fts_index(index.fts_docs)

        _notify_progress(progress_callback, "Refreshing search caches", 0.90)
        index.document_metadata, index.doc_word_counts = build_runtime_search_caches(index.docs)
        _notify_progress(progress_callback, "Deletion complete", 1.0)
    except Exception:
        # Best-effort rollback of the filesystem move if database cleanup fails.
        if moved_to is not None and moved_to.exists() and source_path != moved_to:
            try:
                source_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(moved_to), str(source_path))
            except Exception:
                logger.exception(
                    "Failed to roll back moved file %s to %s after index deletion failure",
                    moved_to,
                    source_path,
                )
        raise

    return {
        "doc_id": int(doc_id),
        "path": str(source_path),
        "deleted_path": str(moved_to) if moved_to is not None else None,
        "word_count": deleted_word_count,
    }


# -----------------------------------------------------------------------------
# Public engine
# -----------------------------------------------------------------------------

def build_runtime_search_caches(
    docs_table: object,
) -> tuple[dict[int, dict], dict[int, int]]:
    """Load only lightweight document metadata needed at startup.

    Positional postings and term/document frequencies stay in LanceDB and are
    fetched only for search candidates. This keeps startup memory/time bounded
    by document and vocabulary metadata rather than the complete occurrence set.
    """

    metadata_rows = query_rows(
        docs_table,
        columns=["id", "path", "sha256", "word_count"],
        operation="Runtime document metadata load",
    )
    document_metadata = {
        int(row["id"]): {
            "id": int(row["id"]),
            "path": str(row["path"]),
            "sha256": str(row["sha256"]),
        }
        for row in metadata_rows
    }
    doc_word_counts = {
        int(row["id"]): int(row.get("word_count", 0) or 0)
        for row in metadata_rows
    }
    return document_metadata, doc_word_counts


@dataclass(slots=True)
class SearchIndex:
    db: object
    docs: object
    postings: object
    vocabulary: object
    frequencies: object
    fts_docs: object
    term_to_id: dict[str, int]
    id_to_term: dict[int, str]
    term_lengths: dict[int, int]
    doc_word_counts: dict[int, int]
    document_metadata: dict[int, dict] = field(default_factory=dict)
    fts_enabled: bool = False


class TextSearchEngine:
    def __init__(
        self,
        db_path: str = DB_PATH,
        search_folder: str = SEARCH_FOLDER,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.search_folder = Path(search_folder).absolute()
        self.search_folder.mkdir(parents=True, exist_ok=True)
        _notify_progress(progress_callback, "Opening database", 0.05)
        self.db = lancedb.connect(str(self.db_path))
        _notify_progress(progress_callback, "Checking database schema", 0.12)
        self.index: SearchIndex | None = self._open_if_valid(
            progress_callback=progress_callback,
        )
        _notify_progress(
            progress_callback,
            "Database ready" if self.index is not None else "No valid index",
            1.0,
        )

    def _open_if_valid(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SearchIndex | None:
        if not index_schema_is_valid(self.db):
            return None

        docs = self.db.open_table(DOCS_TABLE)
        postings = self.db.open_table(POSTINGS_TABLE)
        vocabulary = self.db.open_table(VOCABULARY_TABLE)
        frequencies = self.db.open_table(FREQUENCIES_TABLE)
        fts_docs = self.db.open_table(FTS_TABLE)
        _notify_progress(progress_callback, "Loading vocabulary", 0.25)
        term_to_id, id_to_term, term_lengths = load_vocabulary(vocabulary)
        _notify_progress(progress_callback, "Loading document metadata", 0.75)
        document_metadata, doc_word_counts = build_runtime_search_caches(docs)
        return SearchIndex(
            db=self.db,
            docs=docs,
            postings=postings,
            vocabulary=vocabulary,
            frequencies=frequencies,
            fts_docs=fts_docs,
            term_to_id=term_to_id,
            id_to_term=id_to_term,
            term_lengths=term_lengths,
            doc_word_counts=doc_word_counts,
            document_metadata=document_metadata,
            fts_enabled=True,
        )

    @property
    def has_index(self) -> bool:
        return self.index is not None

    def _refresh_runtime_caches(self) -> None:
        if self.index is None:
            return
        (
            self.index.document_metadata,
            self.index.doc_word_counts,
        ) = build_runtime_search_caches(self.index.docs)

    def _rebuild_sources(self, extra_paths: list[Path]) -> list[Path]:
        sources = list(self.search_folder.glob("*.txt")) + extra_paths
        if self.index is not None:
            for row in query_rows(self.index.docs, columns=["path"]):
                path = Path(row["path"]).absolute()
                if path.exists() and not path_in_folder(path, self.search_folder):
                    sources.append(path)
        return collect_paths(sources)

    def _ensure_ready(
        self,
        extra_paths: list[Path] | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if self.index is not None and index_schema_is_valid(self.db):
            return

        (
            docs,
            postings,
            vocabulary,
            frequencies,
            fts_docs,
            term_to_id,
            fts_enabled,
            doc_word_counts,
        ) = rebuild_index(
            self.db,
            self._rebuild_sources(extra_paths or []),
            progress_callback=progress_callback,
        )
        document_metadata, loaded_word_counts = build_runtime_search_caches(docs)
        self.index = SearchIndex(
            db=self.db,
            docs=docs,
            postings=postings,
            vocabulary=vocabulary,
            frequencies=frequencies,
            fts_docs=fts_docs,
            term_to_id=term_to_id,
            id_to_term={term_id: term for term, term_id in term_to_id.items()},
            term_lengths={term_id: len(term) for term, term_id in term_to_id.items()},
            doc_word_counts=loaded_word_counts or doc_word_counts,
            document_metadata=document_metadata,
            fts_enabled=fts_enabled,
        )

    def index_files(
        self,
        paths: list[str | Path],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        normalized = collect_paths([Path(path) for path in paths])
        if not normalized:
            return {"added": 0, "changed": 0, "removed": 0, "unchanged": 0}
        self._ensure_ready(normalized, progress_callback=progress_callback)
        assert self.index is not None
        return sync_index(
            self.index,
            normalized,
            progress_callback=progress_callback,
        )

    def sync_search_folder(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        current = collect_paths(list(self.search_folder.glob("*.txt")))
        current_strings = {str(path) for path in current}

        if self.index is None or not index_schema_is_valid(self.db):
            sources = self._rebuild_sources([])
            (
                docs,
                postings,
                vocabulary,
                frequencies,
                fts_docs,
                term_to_id,
                fts_enabled,
                doc_word_counts,
            ) = rebuild_index(
                self.db,
                sources,
                progress_callback=progress_callback,
            )
            document_metadata, loaded_word_counts = build_runtime_search_caches(docs)
            self.index = SearchIndex(
                db=self.db,
                docs=docs,
                postings=postings,
                vocabulary=vocabulary,
                frequencies=frequencies,
                fts_docs=fts_docs,
                term_to_id=term_to_id,
                id_to_term={term_id: term for term, term_id in term_to_id.items()},
                term_lengths={term_id: len(term) for term, term_id in term_to_id.items()},
                doc_word_counts=loaded_word_counts or doc_word_counts,
                document_metadata=document_metadata,
                fts_enabled=fts_enabled,
            )
            return {
                "added": len(current),
                "changed": 0,
                "removed": 0,
                "unchanged": 0,
            }

        old = query_rows(self.index.docs, columns=["path"])
        managed_old = {
            str(Path(row["path"]).absolute())
            for row in old
            if path_in_folder(Path(row["path"]).absolute(), self.search_folder)
        }
        remove_paths = managed_old - current_strings
        return sync_index(
            self.index,
            current,
            remove_paths=remove_paths,
            progress_callback=progress_callback,
        )

    def reset_database(
        self,
        action: Literal["delete", "move"],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        """Reset the persistent index and optionally delete/move all text files.

        ``delete`` removes indexed files plus .txt files currently in the search
        folder. ``move`` sends those files to ``search_folder/deleted`` before
        recreating an empty database schema.
        """

        if action not in {"delete", "move"}:
            raise ValueError("Unknown database reset action.")

        search_folder = self.search_folder.absolute()
        deleted_folder = search_folder / "deleted"

        _notify_progress(progress_callback, "Scanning files", 0.05)
        paths: set[Path] = set()
        try:
            paths.update(path.absolute() for path in search_folder.glob("*.txt") if path.is_file())
        except OSError:
            pass

        if self.index is not None:
            for row in query_rows(self.index.docs, columns=["path"]):
                path = Path(str(row["path"])).absolute()
                if path.suffix.casefold() == ".txt" and path.exists():
                    paths.add(path)

        paths = {path for path in paths if path.parent != deleted_folder.absolute()}
        ordered_paths = sorted(paths, key=lambda path: str(path).casefold())
        file_count = len(ordered_paths)

        moved_pairs: list[tuple[Path, Path]] = []
        if file_count:
            for index, source_path in enumerate(ordered_paths, start=1):
                fraction = 0.10 + 0.55 * (index / file_count)
                stage_action = "Deleting files" if action == "delete" else "Moving files to deleted"
                _notify_progress(
                    progress_callback,
                    f"{stage_action} ({index}/{file_count})",
                    fraction,
                )

                if not source_path.exists():
                    continue

                if action == "delete":
                    source_path.unlink()
                else:
                    deleted_folder.mkdir(parents=True, exist_ok=True)
                    destination = _unique_deleted_destination(deleted_folder, source_path)
                    shutil.move(str(source_path), str(destination))
                    moved_pairs.append((source_path, destination))

        _notify_progress(progress_callback, "Clearing database", 0.72)
        try:
            for name in (
                DOCS_TABLE,
                POSTINGS_TABLE,
                *LEGACY_POSTINGS_TABLES,
                VOCABULARY_TABLE,
                FREQUENCIES_TABLE,
                FTS_TABLE,
            ):
                if table_exists(self.db, name):
                    self.db.drop_table(name)

            _notify_progress(progress_callback, "Recreating database", 0.88)
            create_empty_index(self.db)
            self.index = None
            _notify_progress(progress_callback, "Reset complete", 1.0)
        except Exception:
            if action == "move" and moved_pairs:
                for source_path, destination in reversed(moved_pairs):
                    if not destination.exists():
                        continue
                    try:
                        source_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(destination), str(source_path))
                    except Exception:
                        logger.exception(
                            "Failed to restore %s from %s after database reset failure",
                            source_path,
                            destination,
                        )
            raise

        return {
            "file_count": file_count,
            "action": action,
        }

    def count_words_in_file(self, path: str | Path) -> int | None:
        """Return the token count for a readable text file, or ``None`` on failure."""

        try:
            content, _sha256 = read_text_with_sha256(Path(path))
        except Exception:
            logger.exception("Could not count words in %s", path)
            return None
        return count_words(content)

    def list_documents(self) -> list[dict]:
        if self.index is None:
            return []
        rows = query_rows(self.index.docs, columns=["id", "path", "sha256"])
        for row in rows:
            row["word_count"] = int(self.index.doc_word_counts.get(int(row["id"]), 0))
        return sorted(rows, key=lambda row: str(row["path"]).casefold())

    def delete_document(
        self,
        doc_id: int,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        """Delete an indexed document and move its source into ``deleted``."""

        if self.index is None:
            raise ValueError("No index database is available.")
        return delete_document(
            self.index,
            int(doc_id),
            self.search_folder / "deleted",
            progress_callback=progress_callback,
        )

    def search_with_report(
        self,
        terms: list[str],
        mode: SearchMode = "regex",
        *,
        fuzzy_distance: int = FUZZY_DISTANCE,
        max_distance_chars: int = MAX_DISTANCE_CHARS,
        snippet_context_chars: int = SNIPPET_CONTEXT_CHARS,
        max_results: int | None = MAX_RESULTS,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[SearchHit], SearchReport]:
        if self.index is None:
            return [], SearchReport(mode=mode)

        config = SearchConfig(
            fuzzy_distance=fuzzy_distance,
            max_distance_chars=max_distance_chars,
            snippet_context_chars=snippet_context_chars,
            max_proximity_chains_per_document=MAX_PROXIMITY_CHAINS_PER_DOCUMENT,
        )
        return search_index(
            self.index,
            terms,
            mode,
            config,
            max_results,
            progress_callback=progress_callback,
        )

    def search(
        self,
        terms: list[str],
        mode: SearchMode = "regex",
        *,
        fuzzy_distance: int = FUZZY_DISTANCE,
        max_distance_chars: int = MAX_DISTANCE_CHARS,
        snippet_context_chars: int = SNIPPET_CONTEXT_CHARS,
        max_results: int | None = MAX_RESULTS,
        progress_callback: ProgressCallback | None = None,
    ) -> list[SearchHit]:
        hits, _report = self.search_with_report(
            terms,
            mode,
            fuzzy_distance=fuzzy_distance,
            max_distance_chars=max_distance_chars,
            snippet_context_chars=snippet_context_chars,
            max_results=max_results,
            progress_callback=progress_callback,
        )
        return hits


# -----------------------------------------------------------------------------
# Result formatting
# -----------------------------------------------------------------------------

def format_occurrence_count(value: int) -> str:
    """Format an occurrence count compactly for the GUI/result output.

    Counts below 1000 are shown plainly. Larger values use one decimal place
    in scientific notation, e.g. ``1234 -> 1.2e3``.
    """

    value = int(value)
    if value < 1000:
        return str(value)
    mantissa, exponent = f"{float(value):.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_time_report(report: SearchReport) -> list[str]:
    total = max(report.total_ms, 0.000001)

    def stage(label: str, value_ms: float) -> str:
        pct = 100.0 * value_ms / total
        return f"  - {label}: {value_ms:.2f} ms ({pct:.1f}%)"

    lines = [
        "Time report:",
        stage("Total search", report.total_ms),
        stage("Tantivy candidate lookup", report.tantivy_ms),
    ]
    for entry in report.tantivy_by_term:
        lines.append(
            f"    - {entry['query_term']!r}: {entry['candidate_docs']:,} candidate doc(s), "
            f"{entry['time_ms']:.2f} ms [{entry['status']}]"
        )
        if entry.get("query_text") is not None:
            lines.append(f"      FTS query: {entry['query_text']!r}")
        if entry.get("error_type"):
            lines.append(
                f"      Error: {entry['error_type']}: {entry['error_message']}"
            )

    lines.append(
        f"    - keyword candidate intersection: {report.candidate_intersection_ms:.2f} ms "
        f"({report.candidate_docs_before_positions:,} document(s) left)"
    )
    lines.append(stage("Resolve variants", report.variant_resolution_ms))
    if report.mode == "fuzzy":
        length_detail = ""
        if (
            report.variant_candidate_min_length is not None
            and report.variant_candidate_max_length is not None
        ):
            length_detail = (
                f", combined token length {report.variant_candidate_min_length}"
                f"-{report.variant_candidate_max_length}"
            )
        lines.append(
            f"    - in-memory candidate-term enumeration: "
            f"{report.variant_candidate_query_ms:.2f} ms "
            f"({report.variant_candidate_query_rows:,} row(s), "
            f"{report.variant_candidate_term_ids:,} unique term(s){length_detail})"
        )
        lines.append(
            f"    - candidate-term preparation: "
            f"{report.variant_candidate_prepare_ms:.2f} ms"
        )
        lines.append(
            f"    - RapidFuzz total: "
            f"{report.variant_rapidfuzz_ms:.2f} ms"
        )
    for entry in report.variant_resolution:
        extra = f" [{entry['mode_detail']}]" if entry.get("mode_detail") else ""
        lines.append(
            f"    - {entry['query_term']!r}: {entry['variants']:,} variant(s), "
            f"{entry['time_ms']:.2f} ms{extra}"
        )

    lines.append(
        stage("Term-frequency lookup", report.frequency_ms)
        + f" ({report.frequency_rows:,} frequency row(s), "
        f"{report.frequency_docs:,} document(s))"
    )
    lines.append(
        f"    - all-query-term documents: {report.frequency_candidate_docs:,}"
    )
    if report.mode == "fuzzy":
        lines.append(
            "    - fuzzy frequencies are summed across all accepted variants "
            "per original query term"
        )
    lines.append(
        stage("Choose rarest anchor", report.anchor_selection_ms)
        + f" ({len(report.anchor_by_doc):,} document(s))"
    )
    for entry in report.anchor_by_doc[:20]:
        lines.append(
            f"    - doc {entry['doc_id']}: {entry['query_term']!r} "
            f"({entry['occurrences']:,} occurrence(s))"
        )
    if len(report.anchor_by_doc) > 20:
        lines.append(
            f"    - ... {len(report.anchor_by_doc) - 20:,} more document(s)"
        )

    lines.append(
        stage("Fetch positional postings", report.postings_ms)
        + f" ({report.postings_terms:,} term(s), {report.postings_rows:,} posting row(s))"
    )
    lines.append(stage("Build positional matches", report.match_build_ms))
    for entry in report.match_build_by_term:
        lines.append(
            f"    - {entry['query_term']!r}: {entry['occurrences']:,} occurrence(s) "
            f"in {entry['matching_docs']:,} document(s), {entry['time_ms']:.2f} ms"
        )
    lines.append(
        f"    - authoritative document intersection: {report.authoritative_intersection_ms:.2f} ms "
        f"({report.candidate_docs_after_postings:,} document(s) left)"
    )
    lines.append(
        stage("Proximity-chain search", report.proximity_ms)
        + f" ({report.proximity_docs_checked:,} document(s) checked, "
        f"{report.proximity_docs_with_hits:,} with chains, "
        f"max single-document time {report.proximity_max_doc_ms:.2f} ms)"
    )
    lines.append(stage("Fetch result documents", report.documents_ms))
    lines.append(stage("Build result objects", report.result_build_ms))

    lines.extend([
        "",
        "Runtime search instrumentation:",
        f"  - Positional posting rows fetched: {report.runtime_posting_hits:,} across {report.runtime_posting_lookups:,} term/document probe(s)",
        f"  - Frequency rows fetched: {report.runtime_frequency_hits:,} across {report.runtime_frequency_lookups:,} term/document probe(s)",
        f"  - Fuzzy candidate terms enumerated in memory: {report.runtime_candidate_terms:,}",
        f"  - Result source-file reads: {report.runtime_source_file_reads:,}",
        f"  - LanceDB result-document fallbacks: {report.runtime_lancedb_document_fallbacks:,}",
        "",
        "LanceDB/PyArrow query instrumentation (nested inside the stages above):",
        f"  - DB predicate construction (where_in): {report.db_predicate_construction_ms:.2f} ms",
        f"  - table.search()/query setup: {report.db_query_setup_ms:.2f} ms",
        f"  - execution / to_arrow(): {report.db_execution_ms:.2f} ms",
        f"  - Arrow bytes returned: {report.db_bytes_returned:,} bytes",
        f"  - to_pylist() conversion: {report.db_to_pylist_ms:.2f} ms",
        f"  - Python object creation (inclusive in to_pylist()): {report.db_python_object_creation_ms:.2f} ms",
        f"  - Number of DB queries issued: {report.db_queries:,}",
        f"  - Rows returned: {report.db_rows_returned:,}",
        f"  - Bytes returned: {report.db_bytes_returned:,}",
        f"  - Cache probe: {report.db_cache_cold_queries:,} cold / {report.db_cache_warm_queries:,} warm query(es) [process-lifecycle proxy]",
    ])
    if report.db_fragment_file_counts_known:
        lines.append(
            f"  - LanceDB fragment/files touched (runtime metadata): {report.db_fragment_files_touched:,}"
        )
    else:
        lines.append(
            "  - LanceDB fragment/files touched: unavailable from the normal LanceDB/PyArrow query API"
        )
    if report.db_query_breakdown:
        lines.append("  - Per-query DB breakdown:")
        for entry in report.db_query_breakdown:
            fragment = (
                "n/a"
                if entry.get("fragment_files_touched") is None
                else f"{entry['fragment_files_touched']:,}"
            )
            lines.append(
                f"    - {entry['operation']}: {entry['table']}, {entry['cache_state']}, "
                f"setup {entry['setup_ms']:.2f} ms, execution {entry['execution_ms']:.2f} ms, "
                f"to_pylist {entry['to_pylist_ms']:.2f} ms, {entry['rows']:,} row(s), "
                f"{entry['bytes']:,} bytes, fragments/files {fragment}"
            )

    accounted = report.accounted_ms
    overhead = report.total_ms - accounted
    if overhead >= 0:
        lines.append(
            f"  - Uninstrumented overhead: {overhead:.2f} ms "
            f"({100.0 * overhead / total:.1f}%)"
        )
    else:
        lines.append(
            f"  - Timing overlap/measurement variance: {abs(overhead):.2f} ms "
            f"({100.0 * abs(overhead) / total:.1f}%)"
        )
    lines.append(
        f"  - Timed stages total: {accounted:.2f} ms "
        f"({100.0 * accounted / total:.1f}%)"
    )
    if report.diagnostics:
        lines.extend(["", "Diagnostics:"])
        lines.extend(f"  - {diagnostic}" for diagnostic in report.diagnostics)
        lines.append(f"  - Full diagnostic log: {Path(SEARCH_DEBUG_LOG).absolute()}")
    return lines


def format_results(
    hits: list[SearchHit],
    *,
    max_distance_chars: int = MAX_DISTANCE_CHARS,
    snippet_context_chars: int = SNIPPET_CONTEXT_CHARS,
    report: SearchReport | None = None,
) -> str:
    lines = [
        f"Matching documents: {len(hits)}",
        f"Maximum consecutive distance: {max_distance_chars} characters",
        f"Snippet context: {snippet_context_chars} characters",
    ]
    if report is not None:
        lines.extend(["", *format_time_report(report)])
    lines.append("")

    if not hits:
        lines.append("No proximity matches found.")
        return "\n".join(lines).rstrip()

    for number, hit in enumerate(hits, start=1):
        # Very large result sets can spend substantial time building one giant
        # output string. Yield periodically so the GUI thread can repaint the
        # progress ring and remain responsive.
        if number > 1 and number % 16 == 0:
            time.sleep(0)
        document = hit.document
        content = document.get("content", "") or ""
        lines += [
            "=" * 72,
            f"DOCUMENT #{number}",
            "=" * 72,
            f"File: {document.get('path', '')}",
            f"SHA-256: {document.get('sha256', 'n/a')}",
            f"Search mode: {hit.mode}",
            f"Proximity matches: {len(hit.chains)}",
            "Individual term occurrences:",
        ]

        for term_matches in hit.matches_by_term:
            query_term = term_matches[0]["query_term"] if term_matches else ""
            lines.append(
                f"  - '{query_term}': {format_occurrence_count(len(term_matches))} occurrence(s)"
            )

        lines.append("MATCHES:")
        for match_no, chain in enumerate(hit.chains, start=1):
            ordered = sorted(chain, key=lambda m: (m["start"], m["end"]))
            distances = [
                distance_between(ordered[i], ordered[i + 1])
                for i in range(len(ordered) - 1)
            ]
            lines += [
                f"  Match #{match_no}",
                f"  Position: {ordered[0]['start']}-{ordered[-1]['end']}",
                f"  Span: {ordered[-1]['end'] - ordered[0]['start']} characters",
                "  Consecutive distances: "
                + ", ".join(str(d) for d in distances)
                + " characters",
                "  Query terms: " + ", ".join(m["query_term"] for m in ordered),
                "  Matched text: "
                + ", ".join(content[m["start"] : m["end"]] for m in ordered),
                "  Snippet: " + build_snippet(content, ordered, snippet_context_chars),
            ]
            if hit.mode == "fuzzy":
                lines += [
                    "  Indexed terms: " + ", ".join(m["indexed_term"] for m in ordered),
                    "  Edit distances: " + ", ".join(str(m["edit_distance"]) for m in ordered),
                    "  Similarity: " + ", ".join(
                        f"{m['score']:.1f}%" for m in ordered
                    ),
                ]
            lines.append("")

    return "\n".join(lines).rstrip()


__all__ = [
    "DB_PATH",
    "SEARCH_FOLDER",
    "MATCH_MARKER_OPEN",
    "MATCH_MARKER_CLOSE",
    "FUZZY_DISTANCE",
    "MAX_DISTANCE_CHARS",
    "SNIPPET_CONTEXT_CHARS",
    "MAX_RESULTS",
    "FTS_TABLE",
    "SEARCH_DEBUG_LOG",
    "SearchConfig",
    "SearchMode",
    "SearchHit",
    "SearchReport",
    "format_occurrence_count",
    "TextSearchEngine",
    "count_words",
    "find_proximity_chains_numpy",
    "format_results",
    "format_time_report",
]
