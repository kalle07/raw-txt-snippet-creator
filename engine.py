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
    One row per distinct SHA-256 content hash; content is stored once.

DOCUMENT_LOCATIONS
    One row per physical source-file path, linked to one unique content document.

TERMS
    One row per distinct normalized indexed term. ``term_type`` is either
    ``normal`` or ``additional``. ``term_id`` is the numeric key used by all
    persistent term-dependent structures.

POSTINGS
    One row per ``(doc_id, term_id)``. ``positions`` is a flat Arrow
    ``list<int32>`` containing ``[start1, length1, start2, length2, ...]``.

TERM_FREQUENCIES
    One row per ``(doc_id, term_id)`` with only the occurrence count.

FTS_DOCS
    One row per ``(doc_id, term_id)`` containing a parser-safe hexadecimal
    ``fts_key``. Tantivy is used only to accelerate candidate documents;
    authoritative matching uses the unified postings plus the NumPy engine.

Search funnel:
    Python term-type classification and variant resolution
        -> exact Tantivy candidate lookup
        -> term-frequency planning
        -> unified positional postings
        -> NumPy authoritative proximity matching
        -> literal source scan only for unresolved fallback terms
"""

from __future__ import annotations



import hashlib
import json
import inspect
import logging
import os
import re
import unicodedata
import shutil
import threading
import time
import traceback
from datetime import timedelta
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
MAX_RESULTS: int | None = 50
MAX_RESULTS_PER_DOCUMENT: int | None = 20

# Backward-compatible legacy name. New code should use MAX_RESULTS_PER_DOCUMENT.
MAX_PROXIMITY_CHAINS_PER_DOCUMENT: int | None = None

DOCS_TABLE = "docs"
DOCUMENT_LOCATIONS_TABLE = "document_locations"
TERMS_TABLE = "terms"
POSTINGS_TABLE = "postings"
TERM_FREQUENCIES_TABLE = "term_frequencies"
FTS_DOCS_TABLE = "fts_docs"
FTS_KEY_COLUMN = "fts_key"

FTS_MAX_CHUNK_LENGTH = 100
SEARCH_DEBUG_LOG = "search_debug.log"

# LanceDB maintenance policy.
LANCE_OPTIMIZE_AFTER_MODIFICATIONS = 20  # Used only for explicitly requested maintenance.
LANCE_ZERO_AGE = timedelta(0)
LANCE_CLEANUP_EVERY_STARTS = 20
LANCE_CLEANUP_RETENTION = timedelta(days=7)
LANCE_MAINTENANCE_STATE_FILE = ".lancedb_maintenance.json"

# While the app is open, cheaply check directory/file metadata between searches.
# Content hashes are rechecked immediately after a metadata change, and all files
# receive a full SHA-256 verification every five minutes while searches are active.
SEARCH_FOLDER_STAT_CHECK_INTERVAL_SECONDS = 2.0
SEARCH_FOLDER_FULL_HASH_CHECK_INTERVAL_SECONDS = 300.0

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _file_handler = logging.FileHandler(SEARCH_DEBUG_LOG, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_file_handler)

SearchMode = Literal["regex", "fuzzy"]
TermType = Literal["normal", "additional"]
TermKey = tuple[str, str]
QueryRoute = Literal["normal", "additional", "fallback"]
FALLBACK_TERM_ID = -1
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
# old TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_.\-ÄÖÜäöüß]+", re.UNICODE)
# allow Unicode word chars plus underscore, dot, hyphen
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


_STANDARD_TOKEN_RE = re.compile(r"^\w+$", re.UNICODE)

# Characters allowed in additional-tokenizer chunks besides normal word chars.
# `*` and `?` are intentionally excluded because they remain wildcard operators.
ADDITIONAL_TOKEN_SPECIALS = set(
    '!"§$%&/()_=.:;#~><°-—+,'
    "'{}[]@„“|”’‘"
    "\\"
    "“”„«»〝〞〟＂❝❞❠❮❯‟«»"
)

_ADDITIONAL_TOKEN_SPECIALS_ESCAPED = re.escape("".join(ADDITIONAL_TOKEN_SPECIALS))

# Normal word characters used by the additional tokenizer.
_NON_WHITESPACE_TOKEN_RE = re.compile(
    rf"[\w{_ADDITIONAL_TOKEN_SPECIALS_ESCAPED}]+",
    re.UNICODE,
)


_ADDITIONAL_TOKEN_FULL_RE = re.compile(
    rf"^[\w{_ADDITIONAL_TOKEN_SPECIALS_ESCAPED}]+$",
    re.UNICODE,
)


_WORD_CHAR_RE = re.compile(r"\w", re.UNICODE)


_ADDITIONAL_TOKEN_WRAPPER_PAIRS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "«": "»",
    "„": "“",
    "‘": "’",
    "〝": "〞",
    "❮": "❯",
    "❝": "❞",
    '"': '"',    
}


def _is_wrapped_additional_token(token: str) -> bool:
    """
    Return True for chunks wrapped by one balanced punctuation pair.

    Examples:
      (1902/2007)
      [M./Hallwachs]
      {H./Wenning}
      «1902/2007»
    """
    if len(token) < 3:
        return False

    start = token[0]
    end = token[-1]

    if _ADDITIONAL_TOKEN_WRAPPER_PAIRS.get(start) != end:
        return False

    inner = token[1:-1]

    if not inner:
        return False

    # The inner part must still be representable by the additional tokenizer.
    return _ADDITIONAL_TOKEN_FULL_RE.fullmatch(inner) is not None



def iter_additional_tokens(content: str):
    """Yield complete whitespace-delimited chunks with source positions.

    This is the second tokenizer beside ``TOKEN_PATTERN``. It deliberately keeps
    punctuation, slashes, parentheses, dots, hyphens, URLs, names such as
    ``M./Hallwachs`` and date-like strings such as ``(1902/2007)`` intact.
    Positions are retained so the same tokenization can power proximity matching.
    """
    for match in _NON_WHITESPACE_TOKEN_RE.finditer(content):
        raw = match.group(0)
        if raw:
            yield normalize_token(raw), match.start(), match.end()


def encode_literal_fts_key(token: str) -> str:
    """Encode one normalized raw token into a Tantivy-parser-safe exact term."""
    normalized = normalize_token(token)
    return normalized.encode("utf-8").hex()


def _is_normal_token(text: str) -> bool:
    """Return True for plain word/number tokens without punctuation."""
    return bool(_STANDARD_TOKEN_RE.fullmatch(text))

def is_indexable_additional_token(token: str) -> bool:
    """Return True for a canonical additional-token form."""
    if not token or any(char.isspace() for char in token):
        return False
    if len(token) > FTS_MAX_CHUNK_LENGTH:
        return False
    if _STANDARD_TOKEN_RE.fullmatch(token):
        return False
    if _WORD_CHAR_RE.fullmatch(token[0]) and _WORD_CHAR_RE.fullmatch(token[-1]):
        return True
    return _is_wrapped_additional_token(token)



QueryRoute = Literal["normal", "additional", "fallback"]

_EXPLICIT_PATTERN_CHARS = set(r"*?^$+|{}[]\\")

_ADDITIONAL_TRAILING_PUNCTUATION = ",.;!?"


def _strip_additional_wrapper(token: str) -> str:
    """Remove one balanced wrapper pair defined by the wrapper map."""
    if not _is_wrapped_additional_token(token):
        return token
    return token[1:-1]


def canonicalize_additional_token(
    token: str,
    *,
    strip_trailing_punctuation: bool = True,
) -> str:
    """Return the canonical additional term using the configured wrapper pairs.

    Source tokens strip ordinary prose-closing punctuation first. Query terms
    keep such punctuation so forms like ``M./Hormel,`` and ``(1902/2007):``
    remain literal fallbacks. Balanced wrappers themselves are canonicalized
    away, so wrapped and unwrapped forms share one term_id.
    """
    value = normalize_token(token.strip())
    if strip_trailing_punctuation:
        value = value.rstrip(_ADDITIONAL_TRAILING_PUNCTUATION)
    return _strip_additional_wrapper(value)


def classify_query(term: str) -> QueryRoute:
    """Classify one query term before vocabulary resolution."""
    query = normalize_token(term.strip())
    if not query:
        return "fallback"

    # Ordinary closing punctuation stays a literal fallback query. Only the
    # configured balanced wrapper pair is treated as canonical syntax.
    if query.rstrip(_ADDITIONAL_TRAILING_PUNCTUATION) != query:
        return "fallback"

    canonical = canonicalize_additional_token(
        query,
        strip_trailing_punctuation=False,
    )
    if canonical != query and is_indexable_additional_token(canonical):
        return "additional"
    if is_indexable_additional_token(query):
        return "additional"
    if _STANDARD_TOKEN_RE.fullmatch(query):
        return "normal"
    if any(char in _EXPLICIT_PATTERN_CHARS for char in query):
        if re.fullmatch(r"[\w*?^$+|{}\[\]\\.\-]+", query, flags=re.UNICODE):
            return "normal"
    return "fallback"


def _additional_token_core(raw_token: str) -> str:
    """Return canonical source-token text after prose punctuation and wrappers."""
    return canonicalize_additional_token(
        raw_token,
        strip_trailing_punctuation=True,
    )


def find_additional_token_matches(
    content: str,
    query_term: str,
    case_sensitive: bool,
) -> "MatchArrays":
    """Authoritative source fallback using the same wrapper canonicalization."""
    query = canonicalize_additional_token(
        query_term.strip(),
        strip_trailing_punctuation=False,
    )
    query_norm = query if case_sensitive else query.casefold()
    starts: list[int] = []
    ends: list[int] = []

    for _normalized, start, end in iter_additional_tokens(content):
        raw_token = content[int(start):int(end)]
        raw_core = raw_token.rstrip(_ADDITIONAL_TRAILING_PUNCTUATION)
        if not raw_core:
            continue
        canonical = canonicalize_additional_token(
            raw_core,
            strip_trailing_punctuation=False,
        )
        candidate = canonical if case_sensitive else canonical.casefold()
        if candidate != query_norm:
            continue

        span_start = int(start) + (1 if _is_wrapped_additional_token(raw_core) else 0)
        starts.append(span_start)
        ends.append(span_start + len(canonical))

    if not starts:
        return MatchArrays.empty()
    return MatchArrays(
        starts=np.asarray(starts, dtype=np.int32),
        ends=np.asarray(ends, dtype=np.int32),
        term_ids=np.full(len(starts), FALLBACK_TERM_ID, dtype=np.int32),
        edit_distances=np.zeros(len(starts), dtype=np.int32),
    )


def normalize_case_flags(case_sensitive: Sequence[bool] | None, count: int) -> list[bool]:
    if case_sensitive is None:
        return [False] * count
    flags = [bool(value) for value in case_sensitive]
    if len(flags) < count:
        flags.extend([False] * (count - len(flags)))
    return flags[:count]


def find_literal_matches(
    content: str,
    query_term: str,
    case_sensitive: bool,
) -> "MatchArrays":
    """Find literal query text anywhere in the source content.

    This is the final literal fallback.  It intentionally performs a
    substring search rather than requiring complete additional-tokenizer
    chunk equality, so punctuation-bearing queries can match inside a longer
    chunk such as ``H./Timmermann,`` or ``Lichtenstein-Rother/Röbe``.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query_term.strip()), flags)
    matches = list(pattern.finditer(content))
    if not matches:
        return MatchArrays.empty()
    return MatchArrays(
        starts=np.asarray([match.start() for match in matches], dtype=np.int32),
        ends=np.asarray([match.end() for match in matches], dtype=np.int32),
        # ``-1`` identifies a raw-text/literal match; no vocabulary row exists.
        term_ids=np.full(len(matches), FALLBACK_TERM_ID, dtype=np.int32),
        edit_distances=np.zeros(len(matches), dtype=np.int32),
    )


def filter_case_sensitive_matches(
    matches: "MatchArrays",
    content: str,
    query_term: str,
    mode: SearchMode,
    fuzzy_distance: int,
) -> "MatchArrays":
    if len(matches) == 0:
        return matches

    kept_starts: list[int] = []
    kept_ends: list[int] = []
    kept_term_ids: list[int] = []
    kept_distances: list[int] = []
    for start, end, term_id, edit_distance in zip(
        matches.starts.tolist(),
        matches.ends.tolist(),
        matches.term_ids.tolist(),
        matches.edit_distances.tolist(),
    ):
        actual = content[int(start):int(end)]
        if mode == "fuzzy":
            distance = int(Levenshtein.distance(query_term.strip(), actual))
            if distance > fuzzy_distance:
                continue
            edit = distance
        else:
            if "*" in query_term or "?" in query_term:
                pattern = compile_wildcard(query_term, case_sensitive=True)
                if pattern is None or pattern.fullmatch(actual) is None:
                    continue
            elif _has_regex_syntax_except_wildcards(query_term):
                try:
                    if re.fullmatch(query_term, actual) is None:
                        continue
                except re.error:
                    continue
            elif actual != query_term.strip():
                continue
            edit = int(edit_distance)

        kept_starts.append(int(start))
        kept_ends.append(int(end))
        kept_term_ids.append(int(term_id))
        kept_distances.append(int(edit))

    if not kept_starts:
        return MatchArrays.empty()
    return MatchArrays(
        starts=np.asarray(kept_starts, dtype=np.int32),
        ends=np.asarray(kept_ends, dtype=np.int32),
        term_ids=np.asarray(kept_term_ids, dtype=np.int32),
        edit_distances=np.asarray(kept_distances, dtype=np.int32),
    )

# -----------------------------------------------------------------------------
# Persistent schemas
# -----------------------------------------------------------------------------

DOCS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int32()),
        # Compatibility/display path only. The authoritative list of physical
        # paths lives in DOCUMENT_LOCATIONS_TABLE.
        pa.field("path", pa.string()),
        pa.field("sha256", pa.string()),
        pa.field("content", pa.string()),
        pa.field("word_count", pa.int32()),
    ]
)

DOCUMENT_LOCATIONS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int32()),
        pa.field("path", pa.string()),
    ]
)

TERMS_SCHEMA = pa.schema(
    [
        pa.field("term_id", pa.int32()),
        pa.field("term_type", pa.string()),
        pa.field("term", pa.string()),
        pa.field("token_length", pa.int32()),
    ]
)

POSTINGS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int32()),
        pa.field("term_id", pa.int32()),
        pa.field("positions", pa.list_(pa.int32())),
    ]
)

TERM_FREQUENCIES_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int32()),
        pa.field("term_id", pa.int32()),
        pa.field("occurrence_count", pa.int32()),
    ]
)

FTS_DOCS_SCHEMA = pa.schema(
    [
        pa.field("doc_id", pa.int32()),
        pa.field("term_id", pa.int32()),
        pa.field(FTS_KEY_COLUMN, pa.string()),
    ]
)

INDEX_TABLE_SPECS = (
    (DOCS_TABLE, DOCS_SCHEMA),
    (DOCUMENT_LOCATIONS_TABLE, DOCUMENT_LOCATIONS_SCHEMA),
    (TERMS_TABLE, TERMS_SCHEMA),
    (POSTINGS_TABLE, POSTINGS_SCHEMA),
    (TERM_FREQUENCIES_TABLE, TERM_FREQUENCIES_SCHEMA),
    (FTS_DOCS_TABLE, FTS_DOCS_SCHEMA),
)

# -----------------------------------------------------------------------------
# Result/config objects
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SearchConfig:
    fuzzy_distance: int = FUZZY_DISTANCE
    max_distance_chars: int = MAX_DISTANCE_CHARS
    snippet_context_chars: int = SNIPPET_CONTEXT_CHARS
    max_results_per_document: int | None = MAX_RESULTS_PER_DOCUMENT
    # Kept for compatibility with callers that constructed SearchConfig directly.
    # When supplied, this legacy value takes precedence over the new field.
    max_proximity_chains_per_document: int | None = MAX_PROXIMITY_CHAINS_PER_DOCUMENT
    case_sensitive: tuple[bool, ...] = ()

    @property
    def effective_max_results_per_document(self) -> int | None:
        if self.max_proximity_chains_per_document is not None:
            return self.max_proximity_chains_per_document
        return self.max_results_per_document

    def __post_init__(self) -> None:
        if self.fuzzy_distance < 0:
            raise ValueError("FUZZY_DISTANCE must be >= 0")
        if self.max_distance_chars < 0:
            raise ValueError("MAX_DISTANCE_CHARS must be >= 0")
        if self.snippet_context_chars < 0:
            raise ValueError("SNIPPET_CONTEXT_CHARS must be >= 0")

        for name, value in (
            ("MAX_RESULTS_PER_DOCUMENT", self.max_results_per_document),
            (
                "MAX_PROXIMITY_CHAINS_PER_DOCUMENT",
                self.max_proximity_chains_per_document,
            ),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or None")


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
    return all(
        table_exists(db, name) and schema_matches(db.open_table(name), schema)
        for name, schema in INDEX_TABLE_SPECS
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


def scan_search_folder(
    folder: Path,
    excluded_folder: Path,
) -> tuple[list[Path], dict[str, tuple[int, int]]]:
    """Safely enumerate .txt files and capture fast change-detection metadata.

    A traversal/stat error aborts the scan. Callers must never interpret an
    incomplete scan as evidence that indexed documents were deleted.
    """
    folder = folder.absolute()
    excluded_folder = excluded_folder.absolute()
    if not folder.is_dir():
        raise OSError(f"Search folder is not accessible: {folder}")

    paths: list[Path] = []
    snapshot: dict[str, tuple[int, int]] = {}

    def raise_scan_error(error: OSError) -> None:
        raise error

    for root, directory_names, file_names in os.walk(
        folder,
        topdown=True,
        onerror=raise_scan_error,
        followlinks=False,
    ):
        root_path = Path(root).absolute()
        directory_names[:] = [
            name
            for name in directory_names
            if not path_in_folder((root_path / name).absolute(), excluded_folder)
        ]
        for file_name in file_names:
            if Path(file_name).suffix.casefold() != ".txt":
                continue
            path = (root_path / file_name).absolute()
            if path_in_folder(path, excluded_folder):
                continue
            stat_result = path.stat()
            if not path.is_file():
                continue
            paths.append(path)
            snapshot[str(path)] = (
                int(stat_result.st_mtime_ns),
                int(stat_result.st_size),
            )

    return collect_paths(paths), snapshot


def path_in_folder(path: Path, folder: Path) -> bool:
    """Return True when path is inside folder (including folder itself)."""
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False


def iter_indexed_terms(content: str):
    """Yield ``(term_type, normalized_term, start, end)`` for persistent terms."""
    for token, start, end in iter_tokens(content):
        yield "normal", token, int(start), int(end)

    for _normalized_chunk, start, end in iter_additional_tokens(content):
        raw_token = content[int(start):int(end)]
        raw_core = raw_token.rstrip(_ADDITIONAL_TRAILING_PUNCTUATION)
        if not raw_core:
            continue
        canonical = canonicalize_additional_token(raw_core)
        if not canonical or len(canonical) > FTS_MAX_CHUNK_LENGTH:
            continue
        if not is_indexable_additional_token(canonical):
            continue
        span_start = int(start) + (1 if _is_wrapped_additional_token(raw_core) else 0)
        span_end = span_start + len(canonical)
        yield "additional", canonical, span_start, int(span_end)


def indexed_term_keys(content: str) -> set[TermKey]:
    return {
        (str(term_type), str(term))
        for term_type, term, _start, _end in iter_indexed_terms(content)
    }


def assign_term_ids(documents: Sequence[dict]) -> dict[TermKey, int]:
    """Collect all normal/additional terms and assign deterministic IDs."""
    terms: set[TermKey] = set()
    for document in documents:
        terms.update(indexed_term_keys(document["content"]))
    return {term_key: term_id for term_id, term_key in enumerate(sorted(terms))}


def build_postings_for_document(
    doc_id: int,
    content: str,
    term_to_id: dict[TermKey, int],
) -> list[dict]:
    spans_by_term: dict[int, list[int]] = {}
    for term_type, term, start, end in iter_indexed_terms(content):
        term_id = term_to_id[(term_type, term)]
        spans_by_term.setdefault(term_id, []).extend(
            (int(start), int(end - start))
        )

    return [
        {
            "doc_id": int(doc_id),
            "term_id": int(term_id),
            "positions": np.asarray(flat_positions, dtype=np.int32),
        }
        for term_id, flat_positions in sorted(spans_by_term.items())
    ]


def frequency_rows_from_postings(postings: Sequence[dict]) -> list[dict]:
    return [
        {
            "doc_id": int(row["doc_id"]),
            "term_id": int(row["term_id"]),
            "occurrence_count": int(len(row["positions"]) // 2),
        }
        for row in postings
    ]


def fts_document_rows(
    doc_id: int,
    content: str,
    term_to_id: dict[TermKey, int],
) -> list[dict]:
    """Build one exact FTS row per distinct indexed term."""
    keys = indexed_term_keys(content)
    return [
        {
            "doc_id": int(doc_id),
            "term_id": int(term_to_id[(term_type, term)]),
            FTS_KEY_COLUMN: encode_literal_fts_key(term),
        }
        for term_type, term in sorted(
            keys,
            key=lambda key: term_to_id[key],
        )
        if len(term) <= FTS_MAX_CHUNK_LENGTH
    ]


def refresh_fts_index(table: object) -> bool:
    """Create/rebuild the exact Tantivy index over parser-safe hex keys."""
    if not hasattr(table, "create_fts_index"):
        return False

    attempts = (
        {
            "replace": True,
            "base_tokenizer": "raw",
            "with_position": True,
            "stem": False,
            "remove_stop_words": False,
            "ascii_folding": False,
        },
        {
            "replace": True,
            "base_tokenizer": "raw",
            "with_position": True,
        },
        {
            "replace": True,
            "base_tokenizer": "raw",
        },
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            table.create_fts_index(FTS_KEY_COLUMN, **kwargs)
            return True
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc

    logger.error("Could not create exact FTS index: %r", last_error)
    return False



# -----------------------------------------------------------------------------
# FTS/Tantivy candidate filtering
# -----------------------------------------------------------------------------

def _has_regex_syntax_except_wildcards(term: str) -> bool:
    return any(char in "\\.^$+{}[]|()" for char in term)


def fts_query_for_term(term: str) -> str:
    """Return an exact parser-safe Tantivy key."""
    return encode_literal_fts_key(term.strip().casefold())


def all_document_ids(index: "SearchIndex") -> set[int]:
    rows = query_rows(index.docs, columns=["id"], operation="All document IDs")
    return {int(row["id"]) for row in rows}


def _record_fts_query(
    *,
    report: SearchReport,
    index: "SearchIndex",
    search: object,
    started_setup: float,
) -> set[int]:
    setup_ms = (time.perf_counter() - started_setup) * 1000.0
    row_limit = max(int(index.fts_docs.count_rows()), 1)
    started_exec = time.perf_counter()
    arrow, used_score_fallback = _fts_to_arrow_without_score(search, row_limit)
    execution_ms = (time.perf_counter() - started_exec) * 1000.0
    arrow_bytes = int(getattr(arrow, "nbytes", 0) or 0)

    started_py = time.perf_counter()
    rows = arrow.to_pylist()
    to_pylist_ms = (time.perf_counter() - started_py) * 1000.0

    table_key = _table_cache_key(index.fts_docs)
    cache_state = "warm" if table_key in _DB_CACHE_SEEN_TABLES else "cold"
    _DB_CACHE_SEEN_TABLES.add(table_key)
    fragment_file_count = _query_fragment_file_count(search, arrow, index.fts_docs)

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
    report.db_query_breakdown.append(
        {
            "operation": "Tantivy exact candidate lookup",
            "table": table_key,
            "cache_state": cache_state,
            "setup_ms": setup_ms,
            "execution_ms": execution_ms,
            "to_pylist_ms": to_pylist_ms,
            "python_object_creation_ms": to_pylist_ms,
            "rows": len(rows),
            "bytes": arrow_bytes,
            "fragment_files_touched": fragment_file_count,
        }
    )
    if used_score_fallback:
        report.diagnostics.append(
            "LanceDB FTS scanner did not expose disable_scoring_autoprojection; "
            "the score column was projected only as a compatibility fallback."
        )
    return {int(row["doc_id"]) for row in rows}


def fts_candidate_docs(
    index: "SearchIndex",
    query_terms: list[str],
    routes: list[QueryRoute],
    variants_by_term: list[dict[int, int]],
    fallback_flags: list[bool],
    report: SearchReport,
) -> set[int] | None:
    """Use exact candidate acceleration after Python term-type resolution."""
    if not index.fts_enabled:
        return None

    candidate_docs: set[int] | None = None
    all_docs_cache: set[int] | None = None

    def fallback_pool(entry: dict, status: str) -> set[int]:
        nonlocal all_docs_cache
        if all_docs_cache is None:
            all_docs_cache = all_document_ids(index)
        entry["status"] = status
        entry["path"] = "literal fallback candidate pool"
        return set(all_docs_cache)

    for query_term, route, variants, fallback in zip(
        query_terms, routes, variants_by_term, fallback_flags
    ):
        started = time.perf_counter()
        term_type = (
            "additional" if route == "additional"
            else "fallback" if route == "fallback"
            else "normal"
        )
        entry = {
            "query_term": query_term,
            "term_type": term_type,
            "route": route,
            "variants": len(variants),
            "candidate_docs": 0,
            "status": "used",
            "path": "terms/postings",
            "query_text": None,
            "time_ms": 0.0,
        }

        try:
            if fallback or not variants:
                docs_for_keyword = fallback_pool(
                    entry,
                    "literal source-scan fallback",
                )
            elif len(variants) == 1:
                term_id = next(iter(variants))
                key = fts_query_for_term(index.id_to_term[int(term_id)][1])
                entry["query_text"] = key
                setup_started = time.perf_counter()
                search = index.fts_docs.search(key, query_type="fts")
                docs_for_keyword = _record_fts_query(
                    report=report,
                    index=index,
                    search=search,
                    started_setup=setup_started,
                )
                if docs_for_keyword:
                    entry["status"] = "exact"
                    entry["path"] = "Tantivy exact candidate"
                else:
                    docs_for_keyword = fallback_pool(
                        entry,
                        "Tantivy exact lookup returned no candidates; widened",
                    )
            else:
                rows = query_rows_in(
                    index.fts_docs,
                    "term_id",
                    sorted(int(term_id) for term_id in variants),
                    ["doc_id"],
                    operation="Resolved-term candidate filter",
                )
                docs_for_keyword = {int(row["doc_id"]) for row in rows}
                entry["path"] = "LanceDB exact term_id filter"
                entry["status"] = "exact"
                if not docs_for_keyword:
                    docs_for_keyword = fallback_pool(
                        entry,
                        "Resolved-term filter returned no candidates; widened",
                    )
        except Exception as exc:
            docs_for_keyword = fallback_pool(
                entry,
                f"fallback: {type(exc).__name__}",
            )
            entry["error_type"] = type(exc).__name__
            entry["error_message"] = str(exc).strip() or repr(exc)
            report.diagnostics.append(
                f"Exact candidate lookup failed for keyword {query_term!r}: "
                f"{entry['error_type']}: {entry['error_message']}"
            )
            logger.error(
                "Exact candidate lookup failed for keyword=%r\n%s",
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

def load_terms(
    table: object,
) -> tuple[
    dict[int, tuple[str, str]],
    dict[tuple[str, str], int],
    dict[int, int],
    dict[int, list[str]],
    dict[int, list[str]],
]:
    rows = query_rows(
        table,
        columns=["term_id", "term_type", "term", "token_length"],
        operation="Load terms",
    )
    id_to_term: dict[int, tuple[str, str]] = {}
    term_to_id: dict[tuple[str, str], int] = {}
    term_lengths: dict[int, int] = {}
    normal_terms_by_length: dict[int, list[str]] = {}
    additional_terms_by_length: dict[int, list[str]] = {}

    for row in rows:
        term_id = int(row["term_id"])
        term_type = str(row["term_type"])
        term = str(row["term"])
        token_length = int(row["token_length"])
        if term_type not in {"normal", "additional"}:
            raise ValueError(f"Invalid term_type {term_type!r} for term_id {term_id}.")
        id_to_term[term_id] = (term_type, term)
        term_to_id[(term_type, term)] = term_id
        term_lengths[term_id] = token_length
        target = (
            normal_terms_by_length
            if term_type == "normal"
            else additional_terms_by_length
        )
        target.setdefault(token_length, []).append(term)

    for target in (normal_terms_by_length, additional_terms_by_length):
        for length in target:
            target[length].sort()

    return (
        id_to_term,
        term_to_id,
        term_lengths,
        normal_terms_by_length,
        additional_terms_by_length,
    )


def compile_regex(term: str) -> re.Pattern[str] | None:
    try:
        return re.compile(term.casefold())
    except re.error:
        return None


def compile_wildcard(
    term: str,
    *,
    case_sensitive: bool = False,
) -> re.Pattern[str] | None:
    parts: list[str] = []
    source = term if case_sensitive else term.casefold()
    for char in source:
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


def resolve_variants(
    index: "SearchIndex",
    query_term: str,
    term_type: TermType,
    mode: SearchMode,
    config: SearchConfig,
) -> dict[int, int]:
    """Resolve variants strictly within one term type."""
    query = query_term.strip()
    if not query:
        return {}

    normalized = query.casefold()
    term_map = (
        index.normal_terms_by_length
        if term_type == "normal"
        else index.additional_terms_by_length
    )

    if term_type == "additional":
        normalized = canonicalize_additional_token(
            normalized,
            strip_trailing_punctuation=False,
        )
        if not normalized:
            return {}
        if mode == "regex":
            term_id = index.term_to_id.get(("additional", normalized))
            return {int(term_id): 0} if term_id is not None else {}
        if "*" in query or "?" in query:
            return {}

    if term_type == "normal" and mode == "regex":
        if (
            not _has_regex_syntax_except_wildcards(query)
            and "*" not in query
            and "?" not in query
        ):
            term_id = index.term_to_id.get(("normal", normalized))
            return {int(term_id): 0} if term_id is not None else {}

        pattern = (
            compile_wildcard(query)
            if "*" in query or "?" in query
            else compile_regex(query)
        )
        if pattern is None:
            return {}

        result: dict[int, int] = {}
        for terms_at_length in term_map.values():
            for term in terms_at_length:
                if pattern.fullmatch(term):
                    term_id = index.term_to_id[("normal", term)]
                    result[int(term_id)] = 0
        return result

    lower = max(0, len(normalized) - config.fuzzy_distance)
    upper = len(normalized) + config.fuzzy_distance
    candidate_terms: list[str] = []
    candidate_ids: list[int] = []

    candidate_started = time.perf_counter()
    for token_length in range(lower, upper + 1):
        for term in term_map.get(token_length, ()):
            candidate_terms.append(term)
            candidate_ids.append(index.term_to_id[(term_type, term)])
    candidate_enumeration_ms = (time.perf_counter() - candidate_started) * 1000.0

    report = _ACTIVE_DB_QUERY_REPORT.get()
    if report is not None:
        report.variant_candidate_query_ms += candidate_enumeration_ms
        report.variant_candidate_query_rows += len(candidate_terms)
        report.variant_candidate_term_ids += len(candidate_terms)
        report.runtime_candidate_terms += len(candidate_terms)
        if candidate_terms:
            report.variant_candidate_min_length = (
                lower
                if report.variant_candidate_min_length is None
                else min(report.variant_candidate_min_length, lower)
            )
            report.variant_candidate_max_length = (
                upper
                if report.variant_candidate_max_length is None
                else max(report.variant_candidate_max_length, upper)
            )

    if not candidate_terms:
        return {}

    started_fuzzy = time.perf_counter()
    matches = process.extract(
        normalized,
        candidate_terms,
        scorer=Levenshtein.distance,
        score_cutoff=config.fuzzy_distance,
        score_hint=config.fuzzy_distance,
        limit=None,
    )
    fuzzy_ms = (time.perf_counter() - started_fuzzy) * 1000.0
    if report is not None:
        report.variant_rapidfuzz_ms += fuzzy_ms

    return {
        int(candidate_ids[index_in_candidates]): int(distance)
        for _term, distance, index_in_candidates in matches
    }


def _variant_mode_detail(
    route: QueryRoute,
    term_type: TermType,
    mode: SearchMode,
) -> str:
    if route == "fallback":
        return "literal source-scan fallback"
    if term_type == "additional":
        return (
            "additional exact term lookup"
            if mode == "regex"
            else "additional fuzzy term lookup"
        )
    if mode == "fuzzy":
        return "normal fuzzy term lookup"
    if route == "normal":
        return "normal exact/wildcard/regex term lookup"
    return "term lookup"


def resolve_query_terms(
    index: "SearchIndex",
    query_terms: list[str],
    mode: SearchMode,
    config: SearchConfig,
    report: SearchReport,
) -> tuple[list[dict[int, int]], list[QueryRoute], list[bool]]:
    variants_by_term: list[dict[int, int]] = []
    routes: list[QueryRoute] = []
    fallback_flags: list[bool] = []

    started_total = time.perf_counter()
    for query_term in query_terms:
        started = time.perf_counter()
        route = classify_query(query_term)
        routes.append(route)

        if route == "normal":
            term_type: TermType = "normal"
        elif route == "additional":
            term_type = "additional"
        else:
            term_type = "normal"

        if route == "fallback":
            variants = {}
            fallback = True
            detail = "literal source-scan fallback"
        else:
            variants = resolve_variants(
                index,
                query_term,
                term_type,
                mode,
                config,
            )
            fallback = not bool(variants)
            detail = (
                "literal source-scan fallback after unresolved indexed term"
                if fallback
                else _variant_mode_detail(route, term_type, mode)
            )

        variants_by_term.append(variants)
        fallback_flags.append(fallback)
        report.variant_resolution.append(
            {
                "query_term": query_term,
                "term_type": term_type if route != "fallback" else "fallback",
                "route": route,
                "variants": len(variants),
                "path": "literal fallback" if fallback else "terms/postings",
                "mode_detail": detail,
                "time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )

    report.variant_resolution_ms = (
        time.perf_counter() - started_total
    ) * 1000.0
    return variants_by_term, routes, fallback_flags



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

def query_arrow_columns(
    table: object,
    predicate: str | None = None,
    columns: list[str] | None = None,
) -> pa.Table:
    """
    Fetch data and return the raw PyArrow Table without converting to Python lists.
    This is critical for performance when dealing with large nested lists like positions.
    """
    query = table.search()
    if predicate:
        query = query.where(predicate)
    if columns:
        query = query.select(columns)
    
    # Return the Arrow Table directly
    return query.to_arrow()


def fetch_postings(
    index: "SearchIndex",
    term_ids: set[int],
    candidate_docs: set[int],
    report: SearchReport | None = None,
) -> dict[int, dict[int, np.ndarray]]:
    """
    Fetch positional postings using direct Arrow-to-NumPy conversion.
    
    Handles ChunkedArray by combining chunks into a single array before
    extracting the underlying flat values and offsets.
    """
    if not term_ids or not candidate_docs:
        return {}

    grouped: dict[int, dict[int, np.ndarray]] = {}
    candidate_doc_ids = sorted(int(doc_id) for doc_id in candidate_docs)
    stored_term_ids = sorted(int(value) for value in term_ids)
    term_chunk_size = 500

    for offset in range(0, len(stored_term_ids), term_chunk_size):
        term_chunk = stored_term_ids[offset : offset + term_chunk_size]
        
        # Build predicate: doc_id IN (...) AND term_id IN (...)
        doc_predicate = where_in("doc_id", candidate_doc_ids)
        term_predicate = where_in("term_id", term_chunk)
        full_predicate = f"({doc_predicate}) AND ({term_predicate})"

        # Fetch as Arrow Table, NOT Python lists
        arrow_table = query_arrow_columns(
            index.postings,
            predicate=full_predicate,
            columns=["doc_id", "term_id", "positions"],
        )

        if arrow_table.num_rows == 0:
            continue

        # Extract scalar columns as NumPy arrays directly from Arrow
        doc_ids_arr = arrow_table.column("doc_id").to_numpy(zero_copy_only=False)
        term_ids_arr = arrow_table.column("term_id").to_numpy(zero_copy_only=False)
        
        # The positions column is a ChunkedArray of ListArrays.
        # We must combine chunks into a single ListArray to access .values and .offsets.
        pos_chunked = arrow_table.column("positions")
        
        # Combine all chunks into a single contiguous ListArray
        if isinstance(pos_chunked, pa.ChunkedArray):
            pos_list_array = pa.concat_arrays(pos_chunked.chunks)
        else:
            # Fallback if it's already a single array (unlikely from LanceDB)
            pos_list_array = pos_chunked
        
        # Now pos_list_array is a single pa.ListArray
        # Extract the underlying buffers directly to NumPy
        flat_values = pos_list_array.values.to_numpy(zero_copy_only=False).astype(np.int32)
        offsets = pos_list_array.offsets.to_numpy(zero_copy_only=False).astype(np.int32)
        
        # Map each row back to (term_id, doc_id) and slice the flat array
        for i in range(arrow_table.num_rows):
            term_id = int(term_ids_arr[i])
            doc_id = int(doc_ids_arr[i])
            
            start_idx = int(offsets[i])
            # Handle the last row which uses len(flat_values) as end
            end_idx = int(offsets[i + 1]) if i + 1 < len(offsets) else len(flat_values)
            
            # Slice creates a view (zero-copy), very fast
            positions_arr = flat_values[start_idx:end_idx]
            
            grouped.setdefault(term_id, {})[doc_id] = positions_arr

        if report is not None:
            report.runtime_posting_lookups += len(candidate_doc_ids) * len(term_chunk)
            report.runtime_posting_hits += arrow_table.num_rows

    # Ensure all requested term_ids have an entry (even if empty)
    for term_id in stored_term_ids:
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
    id_to_term: dict[int, tuple[str, str]],
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
        if int(term_id) < 0:
            indexed_term = query_term.strip()
        else:
            indexed_term = id_to_term[int(term_id)][1]
        max_len = max(query_len, len(indexed_term))
        score = (
            100.0
            if int(term_id) < 0 or mode == "regex" or max_len == 0
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


def find_proximity_chains_numpy(
    matches_by_term: list[MatchArrays],
    query_terms: list[str],
    id_to_term: dict[int, tuple[str, str]],
    config: SearchConfig,
    anchor_term: object | None = None,
    case_sensitive: Sequence[bool] | None = None,
) -> list[list[dict]]:
    """Find proximity chains using flat NumPy arrays.

    Every query-term occurrence is represented by numeric arrays.  Connected
    components are detected from vectorized gap arithmetic; the final chain is
    materialized back to dictionaries only for the public ``SearchHit`` output.
    """

    if not matches_by_term or any(len(matches) == 0 for matches in matches_by_term):
        return []

    # Required multiplicities are based on normalized user keywords.  Duplicate
    # GUI slots therefore require separate physical occurrences.
    case_flags = normalize_case_flags(case_sensitive, len(query_terms))
    required_counts: dict[tuple[str, bool], int] = {}
    for query_term, sensitive in zip(query_terms, case_flags):
        key = (query_term.casefold(), bool(sensitive))
        required_counts[key] = required_counts.get(key, 0) + 1

    unique_terms = list(required_counts)
    term_code = {term: code for code, term in enumerate(unique_terms)}
    display_terms: dict[tuple[str, bool], str] = {}
    for query_term, sensitive in zip(query_terms, case_flags):
        display_terms.setdefault((query_term.casefold(), bool(sensitive)), query_term)

    starts_parts: list[np.ndarray] = []
    ends_parts: list[np.ndarray] = []
    term_code_parts: list[np.ndarray] = []
    indexed_term_parts: list[np.ndarray] = []
    edit_parts: list[np.ndarray] = []

    for query_term, matches, sensitive in zip(query_terms, matches_by_term, case_flags):
        code = term_code[(query_term.casefold(), bool(sensitive))]
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
        return []

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
        return []

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
        split_points = np.empty(0, dtype=np.int32)

    component_starts = np.concatenate(
        [np.array([0], dtype=np.int32), split_points]
    )
    component_ends = np.concatenate(
        [
            split_points.astype(np.int32, copy=False) - 1,
            np.array([starts.size - 1], dtype=np.int32),
        ]
    )

    anchor_norm = anchor_term if anchor_term in required_counts else None
    if anchor_norm is None:
        # Fallback if the planner did not choose an anchor.
        counts = np.bincount(
            codes.astype(np.int32),
            minlength=len(unique_terms),
        )
        anchor_code = min(
            range(len(unique_terms)),
            key=lambda code: (int(counts[code]), unique_terms[code]),
        )
    else:
        anchor_code = term_code[anchor_norm]

    chains: list[list[dict]] = []

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
            if int(term_id) < 0:
                indexed_term = query_term.strip()
            else:
                indexed_term = id_to_term[int(term_id)][1]
            max_len = max(len(query_term.strip().casefold()), len(indexed_term))
            score = (
                100.0
                if int(term_id) < 0 or max_len == 0
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

    return chains


def rank_and_limit_chains(
    chains: list[list[dict]],
    max_results_per_document: int | None,
) -> tuple[list[list[dict]], tuple[int, int, int]]:
    """Rank chains by match length and return the best per-document subset.

    Chain length is the number of matched occurrences contained in the chain.
    For ties, earlier chains in the document win. The returned document score
    is calculated from the complete chain set before per-document truncation.
    """

    ranked = sorted(
        chains,
        key=lambda chain: (
            -len(chain),
            min((int(match["start"]) for match in chain), default=0),
            max((int(match["end"]) for match in chain), default=0),
        ),
    )
    score = (
        len(ranked[0]) if ranked else 0,
        sum(len(chain) for chain in ranked),
        len(ranked),
    )
    if max_results_per_document is not None:
        ranked = ranked[:max_results_per_document]
    return ranked, score


# -----------------------------------------------------------------------------
# Snippets and result documents
# -----------------------------------------------------------------------------

MATCH_MARKER_OPEN = "\ue000"
MATCH_MARKER_CLOSE = "\ue001"



# but map to U+2022 (bullet) instead of space since we keep formatting intact.
_CONTROL_TO_BULLET = str.maketrans(
    {
        ch: "\u2022"
        for ch in (chr(i) for i in range(0x110000))
        if unicodedata.category(ch) in {"Cc", "Cf", "Cs"}
    }
)


def build_snippet(content: str, chain: list[dict], context_chars: int) -> str:
    """Build a snippet with private match markers for GUI-only highlighting."""
    if not chain:
        return ""

    start = min(m["start"] for m in chain)
    end = max(m["end"] for m in chain)

    snippet_start = max(0, start - context_chars)
    snippet_end = min(len(content), end + context_chars)

    snippet = (
        content[snippet_start:snippet_end]
        .replace("\r", " ")
        .replace("\n", " ")
        .translate(_CONTROL_TO_BULLET)
    )

    parts: list[str] = []
    prev = 0

    for a, b in sorted({(m["start"], m["end"]) for m in chain}):
        if b <= snippet_start or a >= snippet_end:
            continue

        sa = max(snippet_start, a) - snippet_start
        ea = min(snippet_end, b) - snippet_start

        if ea <= sa:
            continue

        parts.append(snippet[prev:sa])
        parts.append(MATCH_MARKER_OPEN)
        parts.append(snippet[sa:ea])
        parts.append(MATCH_MARKER_CLOSE)
        prev = ea

    parts.append(snippet[prev:])
    snippet = "".join(parts)

    prefix = "\n..." if snippet_start else ""
    suffix = "...\n" if snippet_end < len(content) else ""
    return prefix + snippet + suffix




def fetch_documents(
    index: "SearchIndex",
    doc_ids: Sequence[int],
    report: SearchReport | None = None,
) -> dict[int, dict]:
    """Fetch one source text per unique content document and expose all locations."""
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

        expected_hash = str(metadata.get("sha256", ""))
        locations = list(metadata.get("locations", []))
        candidates = locations or [str(metadata.get("path", ""))]
        loaded = False
        last_error: Exception | None = None
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate))
            try:
                content, sha256 = read_text_with_sha256(path)
                if sha256 != expected_hash:
                    raise ValueError("source hash differs from indexed hash")
                document = dict(metadata)
                document["path"] = str(path)
                document["locations"] = locations
                document["content"] = content
                documents[doc_id] = document
                loaded = True
                if report is not None:
                    report.runtime_source_file_reads += 1
                break
            except Exception as exc:
                last_error = exc
        if not loaded:
            logger.warning(
                "Using LanceDB document fallback for doc_id=%s locations=%s: %s",
                doc_id,
                candidates,
                last_error or "no source locations available",
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
        for row in rows:
            doc_id = int(row["id"])
            metadata = index.document_metadata.get(doc_id, {})
            row["locations"] = list(metadata.get("locations", []))
            if row["locations"]:
                row["path"] = row["locations"][0]
            documents[doc_id] = row

    return documents


# -----------------------------------------------------------------------------
# incremental synchronization
# -----------------------------------------------------------------------------


def create_empty_index(db: object):
    """Create the current schema and remove superseded persistent tables."""
    for name, _schema in INDEX_TABLE_SPECS:
        if table_exists(db, name):
            db.drop_table(name)

    return (
        db.create_table(DOCS_TABLE, schema=DOCS_SCHEMA),
        db.create_table(DOCUMENT_LOCATIONS_TABLE, schema=DOCUMENT_LOCATIONS_SCHEMA),
        db.create_table(POSTINGS_TABLE, schema=POSTINGS_SCHEMA),
        db.create_table(TERMS_TABLE, schema=TERMS_SCHEMA),
        db.create_table(TERM_FREQUENCIES_TABLE, schema=TERM_FREQUENCIES_SCHEMA),
        db.create_table(FTS_DOCS_TABLE, schema=FTS_DOCS_SCHEMA),
    )


def rebuild_index(
    db: object,
    paths: list[Path],
    *,
    progress_callback: ProgressCallback | None = None,
):
    """Clean rebuild with one indexed document per distinct SHA-256.

    All source files are read and grouped before any existing table is dropped.
    This protects the previous database from an incomplete/unreadable scan.
    """
    source_paths = collect_paths(paths)
    _notify_progress(progress_callback, "Reading text files", 0.0)

    groups: dict[str, dict[str, object]] = {}
    total_paths = max(len(source_paths), 1)
    for position, path in enumerate(source_paths, start=1):
        try:
            content, sha256 = read_text_with_sha256(path)
        except Exception as exc:
            logger.exception("Could not read %s during rebuild; existing database was not replaced", path)
            raise OSError(f"Could not read text file during index rebuild: {path}") from exc
        group = groups.get(sha256)
        if group is None:
            groups[sha256] = {"content": content, "paths": [path]}
        else:
            # SHA-256 is the content identity. Keep every physical path but store
            # and tokenize the text only once.
            group["paths"].append(path)
        _notify_progress(
            progress_callback,
            "Reading and grouping text files",
            0.35 * position / total_paths,
        )

    documents: list[dict] = []
    location_rows: list[dict] = []
    # Stable IDs across rebuilds: unique hashes are ordered by their first path.
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: str(min(item[1]["paths"], key=lambda value: str(value).casefold())).casefold(),
    )
    for doc_id, (sha256, group) in enumerate(ordered_groups):
        group_paths = sorted(group["paths"], key=lambda value: str(value).casefold())
        representative = group_paths[0]
        content = str(group["content"])
        documents.append(document_row(doc_id, representative, content, sha256))
        location_rows.extend(
            {"doc_id": int(doc_id), "path": str(path.absolute())}
            for path in group_paths
        )

    # Do not destroy a valid old database until all files have been read and
    # their duplicate groups are known.
    (
        docs,
        locations,
        postings_table,
        terms_table,
        frequencies_table,
        fts_table,
    ) = create_empty_index(db)

    _notify_progress(progress_callback, "Building unique-content terms", 0.40)
    term_to_id = assign_term_ids(documents)
    term_rows = [
        {
            "term_id": int(term_id),
            "term_type": str(term_type),
            "term": term,
            "token_length": int(len(term)),
        }
        for (term_type, term), term_id in sorted(
            term_to_id.items(), key=lambda item: item[1]
        )
    ]
    add_rows(terms_table, term_rows, TERMS_SCHEMA)

    postings: list[dict] = []
    frequencies: list[dict] = []
    fts_rows: list[dict] = []
    doc_word_counts: dict[int, int] = {}
    for document in documents:
        doc_id = int(document["id"])
        content = str(document["content"])
        document_postings = build_postings_for_document(doc_id, content, term_to_id)
        postings.extend(document_postings)
        frequencies.extend(frequency_rows_from_postings(document_postings))
        doc_word_counts[doc_id] = int(count_words(content))
        fts_rows.extend(fts_document_rows(doc_id, content, term_to_id))

    _notify_progress(progress_callback, "Writing unique-content index", 0.55)
    add_rows(docs, documents, DOCS_SCHEMA)
    add_rows(locations, location_rows, DOCUMENT_LOCATIONS_SCHEMA)
    add_rows(postings_table, postings, POSTINGS_SCHEMA)
    add_rows(frequencies_table, frequencies, TERM_FREQUENCIES_SCHEMA)
    _notify_progress(progress_callback, "Writing FTS entries", 0.80)
    add_rows(fts_table, fts_rows, FTS_DOCS_SCHEMA)
    _notify_progress(progress_callback, "Creating FTS index", 0.90)
    fts_enabled = refresh_fts_index(fts_table)
    _notify_progress(progress_callback, "Indexing complete", 1.0)

    return (
        docs,
        locations,
        postings_table,
        terms_table,
        frequencies_table,
        fts_table,
        term_to_id,
        fts_enabled,
        doc_word_counts,
    )

def remove_unused_terms(index: "SearchIndex", affected_term_ids: set[int]) -> set[int]:
    """Delete term rows that no longer have any unified postings."""
    if not affected_term_ids:
        return set()

    surviving: set[int] = set()
    ids = sorted(int(value) for value in affected_term_ids)
    for offset in range(0, len(ids), 500):
        rows = query_rows_in(
            index.postings,
            "term_id",
            ids[offset : offset + 500],
            ["term_id"],
            operation="Find surviving term IDs",
        )
        surviving.update(int(row["term_id"]) for row in rows)

    obsolete = affected_term_ids - surviving
    if obsolete:
        delete_rows_in(index.terms, "term_id", sorted(obsolete))
        for term_id in obsolete:
            key = index.id_to_term.pop(int(term_id), None)
            if key is None:
                index.term_lengths.pop(int(term_id), None)
                continue
            index.term_to_id.pop(key, None)
            term_type, term = key
            target = (
                index.normal_terms_by_length
                if term_type == "normal"
                else index.additional_terms_by_length
            )
            values = target.get(len(term))
            if values is not None:
                try:
                    values.remove(term)
                except ValueError:
                    pass
                if not values:
                    target.pop(len(term), None)
            index.term_lengths.pop(int(term_id), None)
    return obsolete



def _cleanup_old_versions_for_table(table: object) -> object | None:
    """Delete superseded Lance table versions and their unreferenced files."""
    method = getattr(table, "cleanup_old_versions", None)
    if not callable(method):
        raise AttributeError("LanceDB table has no cleanup_old_versions() method")

    # Keep a retention window and do not delete unverified files. Cleanup is
    # scheduled infrequently at startup, so immediate/unverified deletion is
    # unnecessary and less conservative for database-file consistency.
    try:
        return method(
            older_than=LANCE_CLEANUP_RETENTION,
            delete_unverified=False,
        )
    except TypeError as first_error:
        try:
            return method(older_than=LANCE_CLEANUP_RETENTION)
        except TypeError:
            try:
                return method()
            except Exception as exc:
                raise RuntimeError(
                    "cleanup_old_versions() could not be executed"
                ) from exc
        except Exception:
            raise
    except Exception:
        raise


def _optimize_table(table: object) -> object | None:
    """Compact a Lance table and prune old versions in the same maintenance pass."""
    method = getattr(table, "optimize", None)
    if not callable(method):
        raise AttributeError("LanceDB table has no optimize() method")

    # cleanup_older_than=0 is deliberate: after the compaction rewrite, the
    # superseded physical files should become immediately eligible for removal.
    try:
        return method(
            cleanup_older_than=LANCE_ZERO_AGE,
            delete_unverified=True,
        )
    except TypeError:
        try:
            return method(cleanup_older_than=LANCE_ZERO_AGE)
        except TypeError as first_error:
            raise RuntimeError(
                "LanceDB optimize() does not expose cleanup_older_than; "
                "refusing to run a potentially non-reclaiming optimization"
            ) from first_error


def _stats_bytes_removed(stats: object | None) -> int:
    try:
        return max(0, int(getattr(stats, "bytes_removed", 0) or 0))
    except Exception:
        return 0


def _maintenance_table_names() -> tuple[str, ...]:
    return tuple(name for name, _schema in INDEX_TABLE_SPECS)


def _compactable_table_names() -> tuple[str, ...]:
    # FTS is rebuilt after document additions/changes; compact the authoritative
    # tables on the normal maintenance cadence.
    return (
        DOCS_TABLE,
        DOCUMENT_LOCATIONS_TABLE,
        TERMS_TABLE,
        POSTINGS_TABLE,
        TERM_FREQUENCIES_TABLE,
    )



def cleanup_lancedb_versions(db: object) -> dict[str, int]:
    """Prune superseded LanceDB versions without compacting live data."""
    removed_bytes = 0
    tables_cleaned = 0
    for name in _maintenance_table_names():
        if not table_exists(db, name):
            continue
        try:
            table = db.open_table(name)
            stats = _cleanup_old_versions_for_table(table)
            removed_bytes += _stats_bytes_removed(stats)
            tables_cleaned += 1
        except Exception:
            logger.exception("LanceDB version cleanup failed for table %s", name)
    return {
        "tables_cleaned": tables_cleaned,
        "bytes_removed": removed_bytes,
    }


def sync_index(
    index: "SearchIndex",
    paths: list[Path],
    *,
    remove_paths: set[str] | None = None,
    trusted_unchanged_paths: set[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Synchronize physical locations and unique content documents.

    Every source path has one row in DOCUMENT_LOCATIONS_TABLE. DOCS, positional
    postings, term frequencies and FTS rows exist only once per distinct SHA-256.
    Existing paths that have a trusted unchanged size/mtime can reuse their
    stored hash. All other scanned paths are read and hashed before writes begin.
    """
    remove_paths = {str(Path(path).absolute()) for path in (remove_paths or set())}
    trusted_unchanged_paths = {
        str(Path(path).absolute()) for path in (trusted_unchanged_paths or set())
    }
    source_paths = collect_paths(paths)
    scanned_path_map = {str(path.absolute()): path for path in source_paths}
    scanned_path_strings = set(scanned_path_map)
    _notify_progress(progress_callback, "Reading and grouping text files", 0.0)

    old_doc_rows = query_rows(
        index.docs,
        columns=["id", "path", "sha256"],
        operation="Load unique indexed documents",
    )
    old_docs_by_id = {int(row["id"]): row for row in old_doc_rows}
    old_docs_by_sha = {str(row["sha256"]): row for row in old_doc_rows}
    old_location_rows = query_rows(
        index.locations,
        columns=["doc_id", "path"],
        operation="Load indexed file locations",
    )
    old_location_by_path = {
        str(Path(row["path"]).absolute()): int(row["doc_id"])
        for row in old_location_rows
    }
    # Preserve every old location that this call was not asked to inspect/remove.
    # index_files() intentionally synchronizes a subset; sync_search_folder() passes
    # all current paths and the missing managed paths as remove_paths.
    desired_location_data: dict[str, dict[str, object]] = {}
    for path_str, old_doc_id in old_location_by_path.items():
        if path_str in remove_paths or path_str in scanned_path_strings:
            continue
        old_doc = old_docs_by_id.get(old_doc_id)
        if old_doc is None:
            raise RuntimeError(f"Location {path_str!r} references missing doc_id {old_doc_id}.")
        desired_location_data[path_str] = {
            "path": Path(path_str),
            "sha256": str(old_doc["sha256"]),
            "content": None,
        }

    scanned_items = list(scanned_path_map.items())
    total_scanned = max(len(scanned_items), 1)
    for position, (path_str, path) in enumerate(scanned_items, start=1):
        old_doc_id = old_location_by_path.get(path_str)
        if old_doc_id is not None and path_str in trusted_unchanged_paths:
            old_doc = old_docs_by_id.get(old_doc_id)
            if old_doc is None:
                raise RuntimeError(f"Location {path_str!r} references missing doc_id {old_doc_id}.")
            content = None
            sha256 = str(old_doc["sha256"])
        else:
            try:
                content, sha256 = read_text_with_sha256(path)
            except Exception as exc:
                logger.exception(
                    "Could not read %s during sync; leaving the index unchanged", path
                )
                raise OSError(f"Could not read text file during synchronization: {path}") from exc
        desired_location_data[path_str] = {
            "path": path,
            "sha256": str(sha256),
            "content": content,
        }
        _notify_progress(
            progress_callback,
            "Reading and grouping text files",
            0.30 * position / total_scanned,
        )

    # Group all retained and scanned physical files by content hash.
    groups: dict[str, dict[str, object]] = {}
    for path_str in sorted(desired_location_data, key=str.casefold):
        record = desired_location_data[path_str]
        sha256 = str(record["sha256"])
        group = groups.get(sha256)
        if group is None:
            groups[sha256] = {
                "paths": [path_str],
                "content": record["content"],
            }
        else:
            group["paths"].append(path_str)
            if group["content"] is None and record["content"] is not None:
                group["content"] = record["content"]

    current_hashes = set(groups)
    # Exact hashes retain their indexed document IDs without touching postings.
    doc_id_by_hash: dict[str, int] = {
        sha256: int(old_doc["id"])
        for sha256, old_doc in old_docs_by_sha.items()
        if sha256 in current_hashes
    }
    assigned_ids = set(doc_id_by_hash.values())
    next_id = max(old_docs_by_id, default=-1) + 1

    # If a path's content changed and its previous hash has no remaining copy,
    # reuse that now-orphaned ID for the new content. It is still fully reindexed.
    reusable_ids_by_hash: dict[str, list[tuple[str, int]]] = {}
    for path_str in sorted(scanned_path_strings, key=str.casefold):
        old_doc_id = old_location_by_path.get(path_str)
        if old_doc_id is None:
            continue
        old_doc = old_docs_by_id.get(old_doc_id)
        current_record = desired_location_data[path_str]
        if old_doc is None:
            continue
        old_sha = str(old_doc["sha256"])
        new_sha = str(current_record["sha256"])
        if old_sha != new_sha and old_sha not in current_hashes and new_sha not in doc_id_by_hash:
            reusable_ids_by_hash.setdefault(new_sha, []).append((path_str, old_doc_id))

    reused_for_new_hash: set[int] = set()
    for sha256 in sorted(
        (value for value in current_hashes if value not in doc_id_by_hash),
        key=lambda value: str(groups[value]["paths"][0]).casefold(),
    ):
        candidates = reusable_ids_by_hash.get(sha256, [])
        reusable_id = next(
            (
                old_id for _path, old_id in candidates
                if old_id not in assigned_ids and old_id not in reused_for_new_hash
            ),
            None,
        )
        if reusable_id is not None:
            doc_id_by_hash[sha256] = int(reusable_id)
            assigned_ids.add(int(reusable_id))
            reused_for_new_hash.add(int(reusable_id))
        else:
            doc_id_by_hash[sha256] = int(next_id)
            assigned_ids.add(int(next_id))
            next_id += 1

    # Determine which unique content documents need a rebuild, which are new,
    # and which no longer have any physical location.
    changed_docs: list[dict] = []
    added_docs: list[dict] = []
    changed_ids: set[int] = set()
    for sha256, group in sorted(
        groups.items(), key=lambda item: str(item[1]["paths"][0]).casefold()
    ):
        doc_id = int(doc_id_by_hash[sha256])
        old_doc = old_docs_by_id.get(doc_id)
        if old_doc is not None and str(old_doc["sha256"]) == sha256:
            continue
        content = group["content"]
        if content is None:
            raise RuntimeError(
                f"Cannot build content hash {sha256}: no source text was read for this new hash."
            )
        representative_path = Path(str(group["paths"][0]))
        new_row = document_row(doc_id, representative_path, str(content), sha256)
        if old_doc is None:
            added_docs.append(new_row)
        else:
            changed_docs.append(new_row)
            changed_ids.add(doc_id)

    assigned_ids = set(doc_id_by_hash.values())
    removed_ids = set(old_docs_by_id) - assigned_ids
    affected_ids = set(removed_ids) | changed_ids

    # Build the exact desired mapping from every physical path to its unique doc.
    desired_path_to_doc = {
        path_str: int(doc_id_by_hash[str(record["sha256"])])
        for path_str, record in desired_location_data.items()
    }
    locations_removed_paths = {
        path_str for path_str, old_id in old_location_by_path.items()
        if path_str not in desired_path_to_doc or desired_path_to_doc[path_str] != old_id
    }
    locations_added = sum(
        1 for path_str, new_id in desired_path_to_doc.items()
        if old_location_by_path.get(path_str) != new_id
    )
    locations_removed = len(locations_removed_paths)
    # A path absent before but carrying an already indexed hash is a copy/rename,
    # not a new indexed content document.
    old_paths_deleted_by_hash: dict[str, int] = {}
    new_paths_added_by_hash: dict[str, int] = {}
    for path_str in locations_removed_paths:
        old_id = old_location_by_path[path_str]
        old_sha = str(old_docs_by_id[old_id]["sha256"])
        if path_str not in desired_path_to_doc:
            old_paths_deleted_by_hash[old_sha] = old_paths_deleted_by_hash.get(old_sha, 0) + 1
    for path_str, new_id in desired_path_to_doc.items():
        if old_location_by_path.get(path_str) != new_id:
            new_sha = str(desired_location_data[path_str]["sha256"])
            if path_str not in old_location_by_path:
                new_paths_added_by_hash[new_sha] = new_paths_added_by_hash.get(new_sha, 0) + 1
    renamed = sum(
        min(count, new_paths_added_by_hash.get(sha, 0))
        for sha, count in old_paths_deleted_by_hash.items()
    )

    stats = {
        "added": len(added_docs),
        "changed": len(changed_docs),
        "removed": len(removed_ids),
        "unchanged": max(0, len(groups) - len(added_docs) - len(changed_docs)),
        "renamed": int(renamed),
        "locations_added": int(locations_added),
        "locations_removed": int(locations_removed),
        "unique_documents": int(len(groups)),
        "physical_files": int(len(desired_path_to_doc)),
    }

    _notify_progress(progress_callback, "Updating locations and unique-content index", 0.38)
    if locations_removed_paths:
        delete_rows_in(index.locations, "path", sorted(locations_removed_paths))
    locations_to_add = [
        {"doc_id": int(doc_id), "path": path_str}
        for path_str, doc_id in sorted(desired_path_to_doc.items(), key=lambda item: item[0].casefold())
        if old_location_by_path.get(path_str) != doc_id
    ]
    add_rows(index.locations, locations_to_add, DOCUMENT_LOCATIONS_SCHEMA)

    affected_term_ids: set[int] = set()
    if affected_ids:
        old_postings = query_rows_in(
            index.postings,
            "doc_id",
            sorted(affected_ids),
            ["term_id"],
            operation="Read terms for removed/changed content documents",
        )
        affected_term_ids.update(int(row["term_id"]) for row in old_postings)
        delete_rows_in(index.postings, "doc_id", sorted(affected_ids))
        delete_rows_in(index.frequencies, "doc_id", sorted(affected_ids))
        delete_rows_in(index.fts_docs, "doc_id", sorted(affected_ids))

    if removed_ids or changed_ids:
        delete_rows_in(index.docs, "id", sorted(removed_ids | changed_ids))
    add_rows(index.docs, changed_docs + added_docs, DOCS_SCHEMA)

    for doc_id in removed_ids:
        index.doc_word_counts.pop(int(doc_id), None)
    for doc_id in changed_ids:
        index.doc_word_counts.pop(int(doc_id), None)

    pending_terms: set[TermKey] = set()
    for document in changed_docs + added_docs:
        pending_terms.update(indexed_term_keys(str(document["content"]))) 

    next_term_id = max(index.term_to_id.values(), default=-1) + 1
    new_terms: list[tuple[TermKey, int]] = []
    for term_key in sorted(pending_terms):
        if term_key in index.term_to_id:
            continue
        term_id = int(next_term_id)
        next_term_id += 1
        index.term_to_id[term_key] = term_id
        index.id_to_term[term_id] = term_key
        index.term_lengths[term_id] = len(term_key[1])
        target = (
            index.normal_terms_by_length
            if term_key[0] == "normal"
            else index.additional_terms_by_length
        )
        target.setdefault(len(term_key[1]), []).append(term_key[1])
        target[len(term_key[1])].sort()
        new_terms.append((term_key, term_id))

    if new_terms:
        add_rows(
            index.terms,
            [
                {
                    "term_id": int(term_id),
                    "term_type": term_key[0],
                    "term": term_key[1],
                    "token_length": int(len(term_key[1])),
                }
                for term_key, term_id in new_terms
            ],
            TERMS_SCHEMA,
        )

    _notify_progress(progress_callback, "Building unique-content postings", 0.55)
    new_postings: list[dict] = []
    new_frequencies: list[dict] = []
    new_fts_rows: list[dict] = []
    for document in changed_docs + added_docs:
        doc_id = int(document["id"])
        content = str(document["content"])
        document_postings = build_postings_for_document(doc_id, content, index.term_to_id)
        new_postings.extend(document_postings)
        new_frequencies.extend(frequency_rows_from_postings(document_postings))
        index.doc_word_counts[doc_id] = int(count_words(content))
        new_fts_rows.extend(fts_document_rows(doc_id, content, index.term_to_id))
        affected_term_ids.update(int(row["term_id"]) for row in document_postings)

    _notify_progress(progress_callback, "Writing unique-content postings", 0.75)
    add_rows(index.postings, new_postings, POSTINGS_SCHEMA)
    add_rows(index.frequencies, new_frequencies, TERM_FREQUENCIES_SCHEMA)
    add_rows(index.fts_docs, new_fts_rows, FTS_DOCS_SCHEMA)
    remove_unused_terms(index, affected_term_ids)

    if changed_docs or added_docs:
        _notify_progress(progress_callback, "Refreshing FTS index", 0.88)
        index.fts_enabled = refresh_fts_index(index.fts_docs)
    if locations_added or locations_removed or removed_ids or changed_docs or added_docs:
        index.document_metadata, index.doc_word_counts = build_runtime_search_caches(
            index.docs, index.locations
        )

    _notify_progress(progress_callback, "Index synchronization complete", 1.0)
    return stats

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
        case_flags = normalize_case_flags(config.case_sensitive, len(query_terms))
        if not query_terms:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            return [], report

        # 1. Python term resolver is authoritative for query variants.
        _notify_progress(progress_callback, "Variant resolution", 0.10)
        variants_by_term, routes, fallback_flags = resolve_query_terms(
            index,
            query_terms,
            mode,
            config,
            report,
        )

        # 2. Tantivy performs exact candidate lookup only. Fuzzy/regex/wildcard
        # semantics have already been resolved in Python against one term type.
        _notify_progress(progress_callback, "Candidate lookup", 0.20)
        candidate_docs = fts_candidate_docs(
            index,
            query_terms,
            routes,
            variants_by_term,
            fallback_flags,
            report,
        )
        if candidate_docs is None:
            candidate_docs = all_document_ids(index)
            report.diagnostics.append(
                f"Tantivy unavailable; using all {len(candidate_docs):,} document(s) as candidates."
            )

        if not candidate_docs:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            _notify_progress(progress_callback, "No results found", 1.0)
            return [], report

        # 3. Frequency planning uses the authoritative frequency table only for
        # resolved indexed terms. Fallback terms remain in the final match
        # intersection and do not get approximated.
        _notify_progress(progress_callback, "Frequency planning", 0.30)
        indexed_terms: list[str] = []
        indexed_variants: list[dict[int, int]] = []
        for query_term, variants, fallback in zip(
            query_terms, variants_by_term, fallback_flags
        ):
            if variants and not fallback:
                indexed_terms.append(query_term)
                indexed_variants.append(variants)

        frequency_started = time.perf_counter()
        if indexed_terms:
            frequency_by_doc_term, frequency_rows = fetch_term_frequencies(
                index,
                indexed_terms,
                indexed_variants,
                candidate_docs,
                report,
            )
            anchor_started = time.perf_counter()
            anchors = choose_anchors(indexed_terms, frequency_by_doc_term)
            report.anchor_selection_ms = (
                time.perf_counter() - anchor_started
            ) * 1000.0
        else:
            frequency_by_doc_term = {}
            frequency_rows = 0
            anchors = {}
            report.anchor_selection_ms = 0.0

        report.frequency_ms = (time.perf_counter() - frequency_started) * 1000.0
        report.frequency_rows = frequency_rows
        report.frequency_docs = len(frequency_by_doc_term)
        report.frequency_candidate_docs = (
            len(anchors) if indexed_terms else len(candidate_docs)
        )
        report.anchor_by_doc = [
            {"doc_id": doc_id, **anchor}
            for doc_id, anchor in sorted(anchors.items())
        ]

        if indexed_terms:
            candidate_docs.intersection_update(anchors)

        report.candidate_docs_before_positions = len(candidate_docs)
        if not candidate_docs:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            _notify_progress(progress_callback, "No results found", 1.0)
            return [], report

        # 4. Unified positional postings. Both normal and additional variants
        # are addressed by the same (doc_id, term_id) lookup.
        required_term_ids = {
            int(term_id)
            for variants in indexed_variants
            for term_id in variants
        }
        postings_started = time.perf_counter()
        postings = fetch_postings(
            index,
            required_term_ids,
            candidate_docs,
            report,
        )
        report.postings_ms = (time.perf_counter() - postings_started) * 1000.0
        report.postings_terms = len(required_term_ids)
        report.postings_rows = sum(len(doc_map) for doc_map in postings.values())
        _notify_progress(progress_callback, "Match building", 0.55)

        needs_source_content = any(fallback_flags) or any(case_flags)
        source_documents: dict[int, dict] = {}
        if needs_source_content:
            source_documents = fetch_documents(
                index,
                sorted(candidate_docs),
                report,
            )

        # 5. Build compact NumPy match arrays for every query slot.
        match_build_started = time.perf_counter()
        match_arrays_by_term: list[dict[int, MatchArrays]] = []

        for slot, (
            query_term,
            route,
            variants,
            fallback,
            case_sensitive,
        ) in enumerate(
            zip(
                query_terms,
                routes,
                variants_by_term,
                fallback_flags,
                case_flags,
            )
        ):
            started = time.perf_counter()

            if fallback or not variants:
                mapping: dict[int, MatchArrays] = {}
                for doc_id in candidate_docs:
                    document = source_documents.get(int(doc_id))
                    if document is None:
                        continue
                    content = str(document.get("content", ""))
                    if route == "additional":
                        matches = find_additional_token_matches(
                            content,
                            query_term,
                            case_sensitive,
                        )
                    else:
                        matches = find_literal_matches(
                            content,
                            query_term,
                            case_sensitive,
                        )
                    if len(matches):
                        mapping[int(doc_id)] = matches
                path_used = "literal fallback"
            else:
                mapping = build_match_arrays(variants, postings)
                path_used = "terms/postings"

                if case_sensitive:
                    filtered_mapping: dict[int, MatchArrays] = {}
                    for doc_id, matches in mapping.items():
                        document = source_documents.get(int(doc_id))
                        if document is None:
                            continue
                        filtered = filter_case_sensitive_matches(
                            matches,
                            str(document.get("content", "")),
                            query_term,
                            mode,
                            config.fuzzy_distance,
                        )
                        if len(filtered):
                            filtered_mapping[int(doc_id)] = filtered
                    mapping = filtered_mapping

            match_arrays_by_term.append(mapping)
            report.match_build_by_term.append(
                {
                    "query_term": query_term,
                    "term_type": (
                        "additional" if route == "additional"
                        else "fallback" if route == "fallback"
                        else "normal"
                    ),
                    "path": path_used,
                    "route": route,
                    "matching_docs": len(mapping),
                    "occurrences": sum(
                        len(items) for items in mapping.values()
                    ),
                    "time_ms": (time.perf_counter() - started) * 1000.0,
                }
            )

        report.match_build_ms = (
            time.perf_counter() - match_build_started
        ) * 1000.0

        # 6. This is the authoritative document intersection.
        intersection_started = time.perf_counter()
        for mapping in match_arrays_by_term:
            candidate_docs.intersection_update(mapping)
            if not candidate_docs:
                break
        report.authoritative_intersection_ms = (
            time.perf_counter() - intersection_started
        ) * 1000.0
        report.candidate_docs_after_postings = len(candidate_docs)

        _notify_progress(progress_callback, "Proximity search", 0.72)
        if not candidate_docs:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            _notify_progress(progress_callback, "No results found", 1.0)
            return [], report

        # 7. Existing NumPy positional/proximity engine remains authoritative.
        hit_chains: dict[int, list[list[dict]]] = {}
        hit_scores: dict[int, tuple[int, int, int]] = {}
        max_results_per_document = config.effective_max_results_per_document
        proximity_started = time.perf_counter()
        for doc_id in sorted(candidate_docs):
            per_term = [
                mapping.get(doc_id, MatchArrays.empty())
                for mapping in match_arrays_by_term
            ]
            doc_started = time.perf_counter()
            anchor_info = anchors.get(doc_id)
            anchor_term = anchor_info["query_term"] if anchor_info else None
            chains = find_proximity_chains_numpy(
                per_term,
                query_terms,
                index.id_to_term,
                config,
                anchor_term=anchor_term,
                case_sensitive=case_flags,
            )
            doc_elapsed_ms = (time.perf_counter() - doc_started) * 1000.0
            report.proximity_docs_checked += 1
            report.proximity_max_doc_ms = max(
                report.proximity_max_doc_ms,
                doc_elapsed_ms,
            )
            if chains:
                report.proximity_docs_with_hits += 1
                ranked_chains, hit_score = rank_and_limit_chains(
                    chains,
                    max_results_per_document,
                )
                hit_chains[doc_id] = ranked_chains
                hit_scores[doc_id] = hit_score

        report.proximity_ms = (
            time.perf_counter() - proximity_started
        ) * 1000.0

        scored_hits = [
            (*hit_scores[doc_id], doc_id)
            for doc_id in hit_chains
        ]
        # Higher maximum chain length ranks first. Total chain length and the
        # total number of chains are deterministic tie-breakers.
        scored_hits.sort(
            key=lambda item: (-item[0], -item[1], -item[2], item[3])
        )
        if max_results is not None:
            scored_hits = scored_hits[:max_results]

        hit_ids = [item[3] for item in scored_hits]
        _notify_progress(progress_callback, "Fetching result documents", 0.88)

        documents_started = time.perf_counter()
        documents = fetch_documents(index, hit_ids, report)
        report.documents_ms = (
            time.perf_counter() - documents_started
        ) * 1000.0

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
                for query_term, mapping in zip(
                    query_terms,
                    match_arrays_by_term,
                )
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

        report.result_build_ms = (
            time.perf_counter() - result_started
        ) * 1000.0
        report.final_documents = len(hits)
        report.total_ms = (
            time.perf_counter() - started_total
        ) * 1000.0

        if not hits:
            report.diagnostics.append("no results found")
            _notify_progress(progress_callback, "No results found", 1.0)
        else:
            _notify_progress(progress_callback, "Search complete", 1.0)
        return hits, report
    finally:
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
    path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
    move_source: bool = True,
    maintenance: bool = True,
) -> dict[str, object]:
    """Remove one physical location; discard indexed content only at its last copy.

    ``path`` identifies the selected copy. For compatibility with callers that
    only pass ``doc_id``, the stored representative path is used when possible,
    otherwise the first current location is selected deterministically.
    """
    doc_id = int(doc_id)
    _notify_progress(progress_callback, "Reading document locations", 0.02)
    doc_rows = query_rows(
        index.docs,
        f"id = {doc_id}",
        ["id", "path", "sha256"],
    )
    if not doc_rows:
        raise ValueError(f"Document id {doc_id} is not indexed.")
    doc_row = doc_rows[0]
    locations = query_rows(
        index.locations,
        f"doc_id = {doc_id}",
        ["doc_id", "path"],
    )
    location_paths = sorted(
        {str(Path(row["path"]).absolute()) for row in locations},
        key=str.casefold,
    )
    if not location_paths:
        # A malformed/orphan document is removed through the usual last-copy path.
        location_paths = [str(Path(doc_row["path"]).absolute())]

    if path is not None:
        source_path = str(Path(path).absolute())
        if source_path not in location_paths:
            raise ValueError(f"Path {source_path!r} is not a location of doc_id {doc_id}.")
    else:
        stored_path = str(Path(doc_row["path"]).absolute())
        source_path = stored_path if stored_path in location_paths else location_paths[0]
    source = Path(source_path)

    deleted_folder = deleted_folder.absolute()
    deleted_folder.mkdir(parents=True, exist_ok=True)
    moved_to: Path | None = None
    if move_source and source.exists():
        _notify_progress(progress_callback, "Moving selected file to deleted", 0.15)
        moved_to = _unique_deleted_destination(deleted_folder, source)
        try:
            shutil.move(str(source), str(moved_to))
        except Exception:
            logger.exception("Could not move %s to %s", source, moved_to)
            raise

    try:
        _notify_progress(progress_callback, "Removing selected file location", 0.30)
        delete_rows_in(index.locations, "path", [source_path])
        remaining_locations = [candidate for candidate in location_paths if candidate != source_path]
        deleted_word_count = int(index.doc_word_counts.get(doc_id, 0))

        if not remaining_locations:
            _notify_progress(progress_callback, "Removing final content index", 0.45)
            affected_term_ids = {
                int(row["term_id"])
                for row in query_rows(
                    index.postings,
                    f"doc_id = {doc_id}",
                    ["term_id"],
                )
            }
            delete_rows_in(index.postings, "doc_id", [doc_id])
            delete_rows_in(index.frequencies, "doc_id", [doc_id])
            delete_rows_in(index.fts_docs, "doc_id", [doc_id])
            delete_rows_in(index.docs, "id", [doc_id])
            remove_unused_terms(index, affected_term_ids)
            index.doc_word_counts.pop(doc_id, None)
        # If other copies still point to the same doc_id, postings, frequencies,
        # FTS rows and the stored content must remain untouched.

        _notify_progress(progress_callback, "Refreshing search caches", 0.85)
        index.document_metadata, index.doc_word_counts = build_runtime_search_caches(
            index.docs, index.locations
        )
        _notify_progress(progress_callback, "Deletion complete", 1.0)
    except Exception:
        if moved_to is not None and moved_to.exists() and source != moved_to:
            try:
                source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(moved_to), str(source))
            except Exception:
                logger.exception(
                    "Failed to roll back moved file %s to %s after index deletion failure",
                    moved_to,
                    source,
                )
        raise

    return {
        "doc_id": doc_id,
        "path": str(source),
        "deleted_path": str(moved_to) if moved_to is not None else None,
        "word_count": deleted_word_count,
        "remaining_locations": len(remaining_locations),
        "content_removed": not remaining_locations,
    }


# -----------------------------------------------------------------------------
# Public engine

# -----------------------------------------------------------------------------
# Public engine
# -----------------------------------------------------------------------------

def build_runtime_search_caches(
    docs_table: object,
    locations_table: object | None = None,
) -> tuple[dict[int, dict], dict[int, int]]:
    """Cache unique-content metadata plus every physical source location."""
    metadata_rows = query_rows(
        docs_table,
        columns=["id", "path", "sha256", "word_count"],
        operation="Runtime unique-document metadata load",
    )
    locations_by_doc: dict[int, list[str]] = {}
    if locations_table is not None:
        location_rows = query_rows(
            locations_table,
            columns=["doc_id", "path"],
            operation="Runtime document-location load",
        )
        for row in location_rows:
            locations_by_doc.setdefault(int(row["doc_id"]), []).append(
                str(Path(row["path"]).absolute())
            )

    document_metadata: dict[int, dict] = {}
    doc_word_counts: dict[int, int] = {}
    for row in metadata_rows:
        doc_id = int(row["id"])
        locations = sorted(set(locations_by_doc.get(doc_id, [])), key=str.casefold)
        stored_path = str(row["path"])
        if not locations and stored_path:
            locations = [stored_path]
        document_metadata[doc_id] = {
            "id": doc_id,
            "path": locations[0] if locations else stored_path,
            "sha256": str(row["sha256"]),
            "locations": locations,
            "location_count": len(locations),
        }
        doc_word_counts[doc_id] = int(row.get("word_count", 0) or 0)
    return document_metadata, doc_word_counts


@dataclass(slots=True)
class SearchIndex:
    db: object
    docs: object
    locations: object
    postings: object
    terms: object
    frequencies: object
    fts_docs: object
    normal_terms_by_length: dict[int, list[str]]
    additional_terms_by_length: dict[int, list[str]]
    term_to_id: dict[tuple[str, str], int]
    id_to_term: dict[int, tuple[str, str]]
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
        self._index_lock = threading.RLock()
        self._folder_snapshot: dict[str, tuple[int, int]] = {}
        self._last_folder_stat_check = 0.0
        self._last_full_integrity_check = 0.0
        _notify_progress(progress_callback, "Opening database", 0.05)
        self.db = lancedb.connect(str(self.db_path))
        self._maintenance_modifications = self._load_maintenance_modifications()
        self._startup_count = self._load_startup_count() + 1
        self._save_maintenance_modifications()

        # Open the persistent tables first, then reconcile source files below.
        # Obsolete-version cleanup still runs only once every 20th startup and
        # is never triggered by ordinary indexing operations.
        _notify_progress(progress_callback, "Checking database schema", 0.12)
        self._ensure_empty_schema_if_missing(progress_callback)
        self.index: SearchIndex | None = self._open_if_valid(
            progress_callback=progress_callback,
        )

        # Opening the app performs a full on-disk hash reconciliation. This
        # catches edits and renames made while the app was closed.
        try:
            _notify_progress(progress_callback, "Synchronizing text folder", 0.82)
            startup_sync = self.sync_search_folder(progress_callback=progress_callback)
            logger.info("Startup text-folder synchronization: %s", startup_sync)
        except Exception:
            # Keep the app usable and preserve the current index if the folder
            # cannot be scanned/read; sync methods fail before treating files as removed.
            logger.exception("Startup text-folder synchronization failed; keeping the loaded index")

        if (
            self.index is not None
            and self._startup_count % LANCE_CLEANUP_EVERY_STARTS == 0
        ):
            _notify_progress(
                progress_callback,
                f"Periodic database cleanup (startup {self._startup_count})",
                0.90,
            )
            try:
                cleanup_stats = cleanup_lancedb_versions(self.db)
                logger.info(
                    "Periodic database cleanup on startup %s: %s table(s), %s byte(s) removed",
                    self._startup_count,
                    cleanup_stats.get("tables_cleaned", 0),
                    cleanup_stats.get("bytes_removed", 0),
                )
            except Exception:
                # A maintenance failure is logged but must not make an otherwise
                # usable search database fail to open.
                logger.exception(
                    "Periodic database cleanup failed on startup %s",
                    self._startup_count,
                )

        _notify_progress(
            progress_callback,
            "Database ready" if self.index is not None else "No valid index",
            1.0,
        )

    @property
    def _maintenance_state_path(self) -> Path:
        return self.db_path / LANCE_MAINTENANCE_STATE_FILE


    def _ensure_empty_schema_if_missing(self, progress_callback: ProgressCallback | None) -> None:
        """Create/migrate the schema without discarding legacy source documents.

        The locations table changes the meaning of DOCS, so an old schema is
        rebuilt from currently available source files. All files are read before
        rebuild_index drops the old tables; a scan/read failure preserves the old DB.
        """
        if index_schema_is_valid(self.db):
            return

        sources: list[Path] = []
        deleted_folder = (self.search_folder / "deleted").absolute()
        # Include all managed files; a failed recursive scan must abort migration.
        current, _snapshot = scan_search_folder(self.search_folder, deleted_folder)
        sources.extend(current)

        # Preserve indexed files outside the managed txt tree where possible.
        if table_exists(self.db, DOCS_TABLE):
            try:
                legacy_rows = query_rows(
                    self.db.open_table(DOCS_TABLE),
                    columns=["path"],
                    operation="Read legacy source paths for deduplication migration",
                )
            except Exception:
                legacy_rows = []
            for row in legacy_rows:
                try:
                    legacy_path = Path(str(row["path"])).absolute()
                except Exception:
                    continue
                if (
                    legacy_path.is_file()
                    and legacy_path.suffix.casefold() == ".txt"
                    and not path_in_folder(legacy_path, deleted_folder)
                ):
                    sources.append(legacy_path)

        _notify_progress(progress_callback, "Migrating to unique-content index", 0.50)
        rebuild_index(self.db, collect_paths(sources), progress_callback=progress_callback)

    def _load_maintenance_modifications(self) -> int:
        try:
            payload = json.loads(self._maintenance_state_path.read_text(encoding="utf-8"))
            value = int(payload.get("logical_modifications", 0))
            return max(0, value)
        except Exception:
            return 0

    def _load_startup_count(self) -> int:
        try:
            payload = json.loads(self._maintenance_state_path.read_text(encoding="utf-8"))
            return max(0, int(payload.get("startup_count", 0)))
        except Exception:
            return 0

    def _save_maintenance_modifications(self) -> None:
        state_path = self._maintenance_state_path
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
            temp_path.write_text(
                json.dumps(
                    {
                        "logical_modifications": int(self._maintenance_modifications),
                        "startup_count": int(getattr(self, "_startup_count", 0)),
                    },
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            temp_path.replace(state_path)
        except Exception:
            logger.exception("Could not persist LanceDB maintenance state")

    def _record_maintenance_modifications(self, count: int) -> None:
        if count <= 0:
            return
        self._maintenance_modifications += int(count)
        self._save_maintenance_modifications()

    @staticmethod
    def _logical_modification_count(result: Mapping[str, object]) -> int:
        return sum(
            max(0, int(result.get(key, 0) or 0))
            for key in (
                "added", "changed", "removed", "renamed",
                "locations_added", "locations_removed",
            )
        )

    def _run_database_maintenance(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        force_if_empty: bool = False,
        reason: str = "database maintenance",
    ) -> dict[str, object]:
        """Run the recommended low-cost cleanup and thresholded compaction policy."""
        del reason  # Kept for GUI/API compatibility; no extra routine log noise.

        if self.index is None:
            return {"cleanup_bytes_removed": 0, "optimized": False}

        _notify_progress(progress_callback, "Cleaning obsolete database data", 0.90)
        cleanup_before = cleanup_lancedb_versions(self.db)

        should_optimize = (
            self._maintenance_modifications >= LANCE_OPTIMIZE_AFTER_MODIFICATIONS
        )
        if force_if_empty:
            try:
                should_optimize = should_optimize or int(self.index.docs.count_rows()) == 0
            except Exception:
                pass

        optimized = False
        optimize_success = True
        optimize_bytes_removed = 0

        if should_optimize:
            _notify_progress(progress_callback, "Optimizing database", 0.95)
            for name in _compactable_table_names():
                if not table_exists(self.db, name):
                    continue
                try:
                    table = self.db.open_table(name)
                    stats = _optimize_table(table)
                    optimize_bytes_removed += _stats_bytes_removed(stats)
                except Exception:
                    optimize_success = False
                    logger.exception("LanceDB optimization failed for table %s", name)

            # Always perform an explicit zero-age cleanup after optimization.
            # This is the final reclaim step for files superseded by compaction.
            cleanup_after = cleanup_lancedb_versions(self.db)
            cleanup_bytes_removed = (
                int(cleanup_before.get("bytes_removed", 0))
                + int(cleanup_after.get("bytes_removed", 0))
            )

            if optimize_success:
                self._maintenance_modifications = 0
                self._save_maintenance_modifications()
                optimized = True
        else:
            cleanup_bytes_removed = int(cleanup_before.get("bytes_removed", 0))

        _notify_progress(progress_callback, "Database maintenance complete", 1.0)
        return {
            "cleanup_bytes_removed": cleanup_bytes_removed,
            "optimized": optimized,
            "optimize_bytes_removed": optimize_bytes_removed,
            "pending_modifications": self._maintenance_modifications,
        }


    def _open_if_valid(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> SearchIndex | None:
        if not index_schema_is_valid(self.db):
            return None

        docs = self.db.open_table(DOCS_TABLE)
        locations = self.db.open_table(DOCUMENT_LOCATIONS_TABLE)
        terms = self.db.open_table(TERMS_TABLE)
        postings = self.db.open_table(POSTINGS_TABLE)
        frequencies = self.db.open_table(TERM_FREQUENCIES_TABLE)
        fts_docs = self.db.open_table(FTS_DOCS_TABLE)

        _notify_progress(progress_callback, "Loading terms", 0.25)
        (
            id_to_term,
            term_to_id,
            term_lengths,
            normal_terms_by_length,
            additional_terms_by_length,
        ) = load_terms(terms)

        _notify_progress(progress_callback, "Loading document metadata", 0.75)
        document_metadata, doc_word_counts = build_runtime_search_caches(docs, locations)
        return SearchIndex(
            db=self.db,
            docs=docs,
            locations=locations,
            postings=postings,
            terms=terms,
            frequencies=frequencies,
            fts_docs=fts_docs,
            normal_terms_by_length=normal_terms_by_length,
            additional_terms_by_length=additional_terms_by_length,
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


    def maintain_storage(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        reason: str = "manual database maintenance",
    ) -> dict[str, object]:
        """Explicitly run storage cleanup and thresholded compaction on request."""
        with self._index_lock:
            return self._run_database_maintenance(
                progress_callback=progress_callback,
                force_if_empty=True,
                reason=reason,
            )

    def _refresh_runtime_caches(self) -> None:
        if self.index is None:
            return
        (
            self.index.document_metadata,
            self.index.doc_word_counts,
        ) = build_runtime_search_caches(self.index.docs, self.index.locations)

    def _rebuild_sources(self, extra_paths: list[Path]) -> list[Path]:
        deleted_folder = (self.search_folder / "deleted").absolute()
        try:
            folder_sources = [
                path.absolute()
                for path in self.search_folder.rglob("*.txt")
                if path.is_file() and not path_in_folder(path.absolute(), deleted_folder)
            ]
        except OSError:
            folder_sources = []

        extra_sources = [
            path.absolute()
            for path in extra_paths
            if not path_in_folder(path.absolute(), deleted_folder)
        ]
        sources = folder_sources + extra_sources
        if self.index is not None:
            for row in query_rows(self.index.locations, columns=["path"]):
                path = Path(row["path"]).absolute()
                if (
                    path.exists()
                    and not path_in_folder(path, self.search_folder)
                    and not path_in_folder(path, deleted_folder)
                ):
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
            locations,
            postings,
            terms,
            frequencies,
            fts_docs,
            _rebuilt_term_to_id,
            fts_enabled,
            doc_word_counts,
        ) = rebuild_index(
            self.db,
            self._rebuild_sources(extra_paths or []),
            progress_callback=progress_callback,
        )

        (
            id_to_term,
            term_to_id,
            term_lengths,
            normal_terms_by_length,
            additional_terms_by_length,
        ) = load_terms(terms)

        document_metadata, loaded_word_counts = build_runtime_search_caches(docs, locations)
        self.index = SearchIndex(
            db=self.db,
            docs=docs,
            locations=locations,
            postings=postings,
            terms=terms,
            frequencies=frequencies,
            fts_docs=fts_docs,
            normal_terms_by_length=normal_terms_by_length,
            additional_terms_by_length=additional_terms_by_length,
            term_to_id=term_to_id,
            id_to_term=id_to_term,
            term_lengths=term_lengths,
            doc_word_counts=loaded_word_counts or doc_word_counts,
            document_metadata=document_metadata,
            fts_enabled=fts_enabled,
        )
        self._maintenance_modifications = 0
        self._save_maintenance_modifications()


    def index_files(
        self,
        paths: list[str | Path],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        with self._index_lock:
            deleted_folder = (self.search_folder / "deleted").absolute()
            normalized = collect_paths(
                [
                    Path(path)
                    for path in paths
                    if not path_in_folder(Path(path).absolute(), deleted_folder)
                ]
            )
            if not normalized:
                return {
                    "added": 0,
                    "changed": 0,
                    "removed": 0,
                    "unchanged": 0,
                    "renamed": 0,
                    "locations_added": 0,
                    "locations_removed": 0,
                    "unique_documents": 0,
                    "physical_files": 0,
                }
            self._ensure_ready(normalized, progress_callback=progress_callback)
            assert self.index is not None
            result = sync_index(
                self.index,
                normalized,
                progress_callback=progress_callback,
            )
            changed = self._logical_modification_count(result)
            self._record_maintenance_modifications(changed)
            return result

    def _sync_scanned_folder(
        self,
        current: list[Path],
        snapshot: dict[str, tuple[int, int]],
        *,
        trusted_unchanged_paths: set[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Reconcile one complete scan, then publish its metadata snapshot."""
        current_strings = {str(path) for path in current}
        full_integrity_check = trusted_unchanged_paths is None or not trusted_unchanged_paths

        if self.index is None or not index_schema_is_valid(self.db):
            self._ensure_ready(current, progress_callback=progress_callback)
            full_integrity_check = True
            unique_count = int(self.index.docs.count_rows()) if self.index is not None else 0
            location_count = int(self.index.locations.count_rows()) if self.index is not None else len(current)
            result = {
                "added": unique_count,
                "changed": 0,
                "removed": 0,
                "unchanged": 0,
                "renamed": 0,
                "locations_added": location_count,
                "locations_removed": 0,
                "unique_documents": unique_count,
                "physical_files": location_count,
            }
        else:
            old = query_rows(self.index.locations, columns=["path"])
            # Every physical path is tracked, including copies that do not match
            # the representative compatibility path stored in DOCS.
            managed_old = {
                str(Path(row["path"]).absolute())
                for row in old
                if path_in_folder(Path(row["path"]).absolute(), self.search_folder)
            }
            remove_paths = managed_old - current_strings
            result = sync_index(
                self.index,
                current,
                remove_paths=remove_paths,
                trusted_unchanged_paths=trusted_unchanged_paths,
                progress_callback=progress_callback,
            )
            changed = self._logical_modification_count(result)
            self._record_maintenance_modifications(changed)

        self._folder_snapshot = dict(snapshot)
        now = time.monotonic()
        self._last_folder_stat_check = now
        if full_integrity_check:
            self._last_full_integrity_check = now
        return result

    def sync_search_folder(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Fully hash and reconcile all .txt files under the search folder.

        The `deleted` subtree is always excluded. A failed/incomplete scan aborts
        before any document is removed from the index.
        """
        with self._index_lock:
            current, snapshot = scan_search_folder(
                self.search_folder,
                self.search_folder / "deleted",
            )
            return self._sync_scanned_folder(
                current,
                snapshot,
                progress_callback=progress_callback,
            )

    def _check_search_folder_before_search(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Check for changes while the app is active without hashing on each query.

        File metadata is scanned at most every couple of seconds. Any metadata or
        path change triggers full SHA-256 reconciliation. A periodic full hash pass
        also catches edits where size/mtime were deliberately preserved.
        """
        if self.index is None:
            return

        now = time.monotonic()
        full_due = (
            self._last_full_integrity_check <= 0.0
            or now - self._last_full_integrity_check
            >= SEARCH_FOLDER_FULL_HASH_CHECK_INTERVAL_SECONDS
        )
        stat_due = (
            self._last_folder_stat_check <= 0.0
            or now - self._last_folder_stat_check
            >= SEARCH_FOLDER_STAT_CHECK_INTERVAL_SECONDS
        )
        if not full_due and not stat_due:
            return

        with self._index_lock:
            now = time.monotonic()
            full_due = (
                self._last_full_integrity_check <= 0.0
                or now - self._last_full_integrity_check
                >= SEARCH_FOLDER_FULL_HASH_CHECK_INTERVAL_SECONDS
            )
            stat_due = (
                self._last_folder_stat_check <= 0.0
                or now - self._last_folder_stat_check
                >= SEARCH_FOLDER_STAT_CHECK_INTERVAL_SECONDS
            )
            if not full_due and not stat_due:
                return
            try:
                current, snapshot = scan_search_folder(
                    self.search_folder,
                    self.search_folder / "deleted",
                )
                self._last_folder_stat_check = time.monotonic()
                if full_due or snapshot != self._folder_snapshot:
                    trusted_unchanged_paths = None
                    if not full_due:
                        trusted_unchanged_paths = {
                            path_str
                            for path_str, metadata in snapshot.items()
                            if self._folder_snapshot.get(path_str) == metadata
                        }
                    self._sync_scanned_folder(
                        current,
                        snapshot,
                        trusted_unchanged_paths=trusted_unchanged_paths,
                        progress_callback=progress_callback,
                    )
                else:
                    self._folder_snapshot = dict(snapshot)
            except Exception:
                # A query should still be allowed with the last known-good index.
                # Failed scans/syncs must not advance the trusted snapshot or
                # remove documents from the persistent index.
                self._last_folder_stat_check = time.monotonic()
                logger.exception(
                    "Automatic text-folder check failed; search will use the last known-good index"
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

        with self._index_lock:
            if action not in {"delete", "move"}:
                raise ValueError("Unknown database reset action.")

            search_folder = self.search_folder.absolute()
            deleted_folder = search_folder / "deleted"

            _notify_progress(progress_callback, "Scanning files", 0.05)
            paths: set[Path] = set()
            try:
                paths.update(
                    path.absolute()
                    for path in search_folder.rglob("*.txt")
                    if path.is_file() and not path_in_folder(path.absolute(), deleted_folder)
                )
            except OSError:
                pass

            if self.index is not None:
                for row in query_rows(self.index.locations, columns=["path"]):
                    path = Path(str(row["path"])).absolute()
                    if (
                        path.suffix.casefold() == ".txt"
                        and path.exists()
                        and not path_in_folder(path, deleted_folder)
                    ):
                        paths.add(path)

            paths = {
                path
                for path in paths
                if not path_in_folder(path, deleted_folder)
            }
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
                for name, _schema in INDEX_TABLE_SPECS:
                    if table_exists(self.db, name):
                        self.db.drop_table(name)

                _notify_progress(progress_callback, "Recreating database", 0.88)
                reset_tables = create_empty_index(self.db)
                del reset_tables
                self.index = None
                self._maintenance_modifications = 0
                self._save_maintenance_modifications()
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
        """List every physical file, while exposing its shared content doc ID."""
        with self._index_lock:
            if self.index is None:
                return []
            doc_rows = query_rows(
                self.index.docs,
                columns=["id", "path", "sha256", "word_count"],
                operation="List unique indexed documents",
            )
            docs_by_id = {int(row["id"]): row for row in doc_rows}
            paths_by_doc: dict[int, list[str]] = {}
            for loc in query_rows(
                self.index.locations,
                columns=["doc_id", "path"],
                operation="List document locations",
            ):
                paths_by_doc.setdefault(int(loc["doc_id"]), []).append(
                    str(Path(loc["path"]).absolute())
                )

            rows: list[dict] = []
            for doc_id, locations in paths_by_doc.items():
                document = docs_by_id.get(doc_id)
                if document is None:
                    continue
                ordered_locations = sorted(set(locations), key=str.casefold)
                for location_path in ordered_locations:
                    rows.append({
                        "id": doc_id,
                        "doc_id": doc_id,
                        "path": location_path,
                        "sha256": str(document["sha256"]),
                        "word_count": int(self.index.doc_word_counts.get(doc_id, document.get("word_count", 0)) or 0),
                        "copy_count": len(ordered_locations),
                        "locations": list(ordered_locations),
                    })
            return sorted(rows, key=lambda row: str(row["path"]).casefold())

    def delete_document(
        self,
        doc_id: int,
        *,
        path: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
        move_source: bool = True,
        maintenance: bool = False,
    ) -> dict[str, object]:
        """Delete an indexed document, optionally moving its source into ``deleted``."""

        with self._index_lock:
            if self.index is None:
                raise ValueError("No index database is available.")
            result = delete_document(
                self.index,
                int(doc_id),
                self.search_folder / "deleted",
                path=path,
                progress_callback=progress_callback,
                move_source=move_source,
                maintenance=maintenance,
            )
            self._record_maintenance_modifications(1)
            if maintenance:
                self._run_database_maintenance(
                    progress_callback=progress_callback,
                    force_if_empty=True,
                    reason="document deletion",
                )
            return result

    def search_with_report(
        self,
        terms: list[str],
        mode: SearchMode = "regex",
        *,
        fuzzy_distance: int = FUZZY_DISTANCE,
        max_distance_chars: int = MAX_DISTANCE_CHARS,
        snippet_context_chars: int = SNIPPET_CONTEXT_CHARS,
        max_results: int | None = MAX_RESULTS,
        max_results_per_document: int | None = MAX_RESULTS_PER_DOCUMENT,
        case_sensitive: Sequence[bool] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[SearchHit], SearchReport]:
        with self._index_lock:
            self._check_search_folder_before_search(progress_callback=progress_callback)
            if self.index is None:
                return [], SearchReport(mode=mode)

            raw_case_flags = normalize_case_flags(case_sensitive, len(terms))
            clean_terms: list[str] = []
            clean_case_flags: list[bool] = []
            for term, sensitive in zip(terms, raw_case_flags):
                cleaned = term.strip()
                if not cleaned:
                    continue
                clean_terms.append(cleaned)
                clean_case_flags.append(bool(sensitive))

            config = SearchConfig(
                fuzzy_distance=fuzzy_distance,
                max_distance_chars=max_distance_chars,
                snippet_context_chars=snippet_context_chars,
                max_results_per_document=max_results_per_document,
                case_sensitive=tuple(clean_case_flags),
            )
            return search_index(
                self.index,
                clean_terms,
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
        max_results_per_document: int | None = MAX_RESULTS_PER_DOCUMENT,
        case_sensitive: Sequence[bool] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[SearchHit]:
        hits, _report = self.search_with_report(
            terms,
            mode,
            fuzzy_distance=fuzzy_distance,
            max_distance_chars=max_distance_chars,
            snippet_context_chars=snippet_context_chars,
            max_results=max_results,
            max_results_per_document=max_results_per_document,
            case_sensitive=case_sensitive,
            progress_callback=progress_callback,
        )
        return hits


# -----------------------------------------------------------------------------
# Result formatting
# -----------------------------------------------------------------------------

def _notepadpp_offset(content: str, char_position: int) -> int:
    """Return the UTF-8 byte offset used by Scintilla/Notepad++ positions.

    Internal match coordinates are Python string offsets (Unicode code points),
    but Notepad++ stores document positions in its UTF-8 Scintilla buffer.
    Encoding the prefix therefore also counts CRLF as two bytes and multi-byte
    UTF-8 characters according to the editor's actual document coordinates.
    """
    position = max(0, min(int(char_position), len(content)))
    return len(content[:position].encode("utf-8"))


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
            f"{entry['time_ms']:.2f} ms [{entry['status']}] "
            f"type={entry.get('term_type', 'n/a')} "
            f"path={entry.get('path', 'n/a')}"
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
            f"{entry['time_ms']:.2f} ms [{entry.get('path', 'n/a')}] {extra}"
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
            f"in {entry['matching_docs']:,} document(s), {entry['time_ms']:.2f} ms "
            f"[type={entry.get('term_type', 'n/a')}] [{entry.get('path', 'n/a')}]"
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
        lines.append("No results found.")
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
            f"Physical copies: {len(document.get('locations', []) or [document.get('path', '')])}",
        ]
        for location in document.get("locations", []) or [document.get("path", "")]:
            lines.append(f"  Location: {location}")
        lines += [
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
            start_position = _notepadpp_offset(content, ordered[0]["start"])
            end_position = _notepadpp_offset(content, ordered[-1]["end"])
            lines += [
                f"  Match #{match_no}",
                f"  Position: {start_position}-{end_position}",
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
    "MAX_RESULTS_PER_DOCUMENT",
    "TERMS_TABLE",
    "DOCUMENT_LOCATIONS_TABLE",
    "POSTINGS_TABLE",
    "TERM_FREQUENCIES_TABLE",
    "FTS_DOCS_TABLE",
    "FTS_MAX_CHUNK_LENGTH",
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
