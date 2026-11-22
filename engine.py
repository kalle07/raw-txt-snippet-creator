
"""Persistent positional text index and search engine.

Dependencies:
    lancedb, numpy, pyarrow, psutil, rapidfuzz

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
    ``wrapper_flags`` is a parallel ``list<string>`` with one entry per
    occurrence aligned to positions: ``""`` marks an unwrapped occurrence,
    while values such as ``"()"``, ``"[]"``, ``"«»"`` record which wrapper
    pair surrounded that occurrence.

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
        -> database postings only (unresolved terms return no matches)

Literal text mode is a separate execution path that bypasses index lookup.
"""

from __future__ import annotations

import concurrent.futures
from array import array
from collections import OrderedDict
import hashlib
import json
import inspect
import os
import re
import unicodedata
import shutil
import stat as stat_module
import threading
import time
import traceback
from bisect import bisect_left
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Sequence

from app_logging import get_app_logger

import lancedb
import numpy as np
import psutil
import pyarrow as pa
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

# Module map:
# Configuration/constants -> schemas/models -> tokenization -> source scanning ->
# LanceDB primitives -> index construction -> candidate retrieval -> query planning ->
# positional matching -> snippets/documents -> maintenance -> synchronization ->
# search facade -> report formatting.
# Definitions stay module-level to preserve the GUI and integration import API.

# =============================================================================
# Configuration and compatibility constants
# =============================================================================

DB_PATH = "text_search.lancedb"

SEARCH_FOLDER = "txt"

FUZZY_DISTANCE = 1

MAX_DISTANCE_CHARS = 500

SNIPPET_CONTEXT_CHARS = 200

MAX_RESULTS: int | None = 50

MAX_RESULTS_PER_DOCUMENT: int | None = 20

MAX_PROXIMITY_CHAINS_PER_DOCUMENT: int | None = None

DOCS_TABLE = "docs"

DOCUMENT_LOCATIONS_TABLE = "document_locations"

TERMS_TABLE = "terms"

POSTINGS_TABLE = "postings"

TERM_FREQUENCIES_TABLE = "term_frequencies"

FTS_DOCS_TABLE = "fts_docs"

FTS_KEY_COLUMN = "fts_key"

FTS_MAX_CHUNK_LENGTH = 100

LANCE_OPTIMIZE_AFTER_MODIFICATIONS = 20  # Used only for explicitly requested maintenance.

LANCE_ZERO_AGE = timedelta(0)

LANCE_CLEANUP_EVERY_STARTS = 20

LANCE_CLEANUP_RETENTION = timedelta(days=7)

LANCE_MAINTENANCE_STATE_FILE = ".lancedb_maintenance.json"
SOURCE_MANIFEST_FILE = ".source_manifest.json"
SOURCE_MANIFEST_VERSION = 1
SOURCE_MANIFEST_FULL_VERIFY_EVERY_STARTS = 20

# Decoded positional rows are stable between searches until indexing changes
# the database. Reuse them across searches, but keep retained memory bounded.
POSTING_CACHE_MAX_BYTES = 64 * 1024 * 1024
POSTING_CACHE_MAX_ENTRIES = 150_000

# Integer scalar indexes support the equality/IN filters used by positional
# and frequency lookups. Creation is best-effort for older LanceDB versions.
SEARCH_SCALAR_INDEXES = {
    POSTINGS_TABLE: ("doc_id", "term_id"),
    TERM_FREQUENCIES_TABLE: ("doc_id", "term_id"),
}

# Measured fastest configuration for file-level tokenization. The worker count
# is capped by available CPUs and the number of unique documents.
TOKENIZATION_BACKEND: Literal["processes", "threads"] = "processes"
TOKENIZATION_MAX_WORKERS = 8
TOKENIZATION_BATCH_SIZE = 1

SEARCH_DEBUG_LOG = "search_debug.log"

logger = get_app_logger("engine")

SearchMode = Literal["regex", "fuzzy"]

TermSearchMode = Literal["regex", "fuzzy", "both"]

TermType = Literal["normal", "additional"]

TermKey = tuple[str, str]

QueryRoute = Literal["normal", "additional", "unindexed", "wildcard"]

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
        logger.warning("Progress callback failed at stage %r", stage, exc_info=True)


def get_physical_core_count() -> int:
    """Return the physical CPU-core count reported by psutil.

    If the platform cannot report physical cores, fall back to one worker rather
    than using logical CPUs and potentially oversubscribing the machine.
    """
    try:
        count = psutil.cpu_count(logical=False)
    except (OSError, RuntimeError) as exc:
        logger.warning("Could not detect physical CPU cores with psutil: %s", exc)
        count = None

    if count is None or int(count) < 1:
        logger.warning(
            "psutil did not report physical CPU cores; limiting tokenization to one worker"
        )
        return 1
    return int(count)


def get_tokenization_worker_count(
    task_count: int,
    requested_workers: int = TOKENIZATION_MAX_WORKERS,
) -> int:
    """Cap tokenization workers by the configured limit and physical core count."""
    if task_count <= 0:
        return 0
    return min(
        max(1, int(requested_workers)),
        get_physical_core_count(),
        int(task_count),
    )

# =============================================================================
# Persistent schemas and result models
# =============================================================================

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
        pa.field("wrapper_flags", pa.list_(pa.string())),
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
    term_modes: tuple[TermSearchMode, ...] = ()
    literal_text_search: bool = False

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
    mode: str
    term_modes: list[str] = field(default_factory=list)
    # One entry per query slot, aligned with ``matches_by_term``. Each item
    # contains the source span and its exact wrapper annotation (``""`` means
    # the occurrence was not wrapped).
    wrapper_annotations: list[list[dict]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class PhraseComponent:
    """One token or positional-wildcard component in a phrase."""

    query_term: str
    query_start: int
    query_end: int
    required_wrapper: str | None = None
    # A component made exclusively from * and ? is resolved directly against
    # source token positions. It must never be expanded into vocabulary terms.
    wildcard_pattern: str | None = None


@dataclass(frozen=True, slots=True)
class PhraseSpec:
    """Indexed phrase components plus exact punctuation/whitespace boundaries."""

    components: tuple[PhraseComponent, ...]
    gap_patterns: tuple[re.Pattern[str], ...]
    gap_lengths: tuple[int, ...]
    prefix_pattern: re.Pattern[str] | None
    prefix_length: int
    suffix_pattern: re.Pattern[str] | None
    suffix_length: int


_ACTIVE_DB_QUERY_REPORT: ContextVar["SearchReport | None"] = ContextVar(
    "active_db_query_report", default=None
)

_DB_CACHE_SEEN_TABLES: set[str] = set()


@dataclass(slots=True)
class SearchReport:
    """Detailed timing and candidate statistics for one search execution."""

    mode: str
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
    wildcard_candidate_terms_examined: int = 0
    wildcard_candidate_terms_matched: int = 0
    wildcard_candidate_lookup_ms: float = 0.0
    variant_resolution: list[dict] = field(default_factory=list)
    tantivy_ms: float = 0.0
    tantivy_by_term: list[dict] = field(default_factory=list)
    candidate_intersection_ms: float = 0.0
    candidate_docs_after_fts: int = 0
    candidate_docs_before_positions: int = 0
    selection_applied: bool = False
    selection_filter_ms: float = 0.0
    selected_doc_id_count: int = 0
    candidate_docs_before_selection: int = 0
    candidate_docs_after_selection: int = 0
    phrase_prefilter_ms: float = 0.0
    phrase_docs_before_prefilter: int = 0
    phrase_docs_after_prefilter: int = 0
    source_text_ms: float = 0.0
    source_text_docs: int = 0
    frequency_ms: float = 0.0
    frequency_rows: int = 0
    frequency_docs: int = 0
    frequency_candidate_docs: int = 0
    anchor_selection_ms: float = 0.0
    anchor_by_doc: list[dict] = field(default_factory=list)
    anchor_gate_ms: float = 0.0
    anchor_docs_before_gate: int = 0
    anchor_docs_after_gate: int = 0
    anchor_component_builds: int = 0
    postings_ms: float = 0.0
    postings_terms: int = 0
    postings_rows: int = 0
    posting_cache_hits: int = 0
    posting_cache_misses: int = 0
    posting_cache_bytes: int = 0
    posting_cache_entries: int = 0
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
    db_metadata_ms: float = 0.0
    db_metadata_calls: int = 0
    db_metadata_breakdown: list[dict] = field(default_factory=list)
    db_query_setup_ms: float = 0.0
    db_execution_ms: float = 0.0
    db_to_pylist_ms: float = 0.0
    db_nested_to_pylist_ms: float = 0.0
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
    runtime_source_bytes_read: int = 0
    runtime_lancedb_document_fallbacks: int = 0

    # Literal-mode wall-clock and per-worker instrumentation. Worker durations
    # are summed CPU-thread elapsed times and can exceed the parallel wall time.
    literal_document_list_ms: float = 0.0
    literal_query_prepare_ms: float = 0.0
    literal_parallel_wall_ms: float = 0.0
    literal_parallel_process_cpu_ms: float = 0.0
    literal_thread_cpu_ms: float = 0.0
    literal_byte_normalize_ms: float = 0.0
    literal_byte_match_ms: float = 0.0
    literal_decode_ms: float = 0.0
    literal_bytes_searched: int = 0
    literal_byte_search_docs: int = 0
    literal_decoded_docs: int = 0
    # Raw occurrence statistics are collected before proximity filtering.
    # Per-term counters distinguish document coverage from occurrence count.
    literal_raw_term_stats: list[dict] = field(default_factory=list)
    literal_raw_occurrences_total: int = 0
    literal_raw_count_ms: float = 0.0
    literal_raw_summary_ms: float = 0.0
    literal_raw_candidate_docs: int = 0
    literal_raw_candidate_occurrences: int = 0
    literal_raw_candidate_max_occurrences: int = 0
    literal_raw_candidate_max_doc_id: int | None = None
    literal_raw_candidate_top_docs: list[dict] = field(default_factory=list)
    literal_docs_before_limit: int = 0
    literal_docs_after_limit: int = 0
    literal_sort_limit_ms: float = 0.0
    literal_result_assembly_ms: float = 0.0
    literal_batch_count: int = 0
    literal_worker_count: int = 0
    literal_batch_size: int = 0
    literal_worker_batches: list[dict] = field(default_factory=list)

    diagnostics: list[str] = field(default_factory=list)

    @property
    def accounted_ms(self) -> float:
        return sum(
            (
                self.variant_resolution_ms,
                self.tantivy_ms,
                self.candidate_intersection_ms,
                self.selection_filter_ms,
                self.frequency_ms,
                self.anchor_selection_ms,
                self.postings_ms,
                self.anchor_gate_ms,
                self.match_build_ms,
                self.authoritative_intersection_ms,
                self.proximity_ms,
                self.source_text_ms,
                self.documents_ms,
                self.result_build_ms,
            )
        )


@dataclass(slots=True)
class PostingOccurrences:
    """One document/term posting with occurrence-aligned source annotations."""

    positions: np.ndarray
    wrapper_flags: np.ndarray


class PostingLRUCache:
    """Thread-safe, byte-bounded LRU for decoded ``(doc_id, term_id)`` rows."""

    def __init__(
        self,
        max_bytes: int = POSTING_CACHE_MAX_BYTES,
        max_entries: int = POSTING_CACHE_MAX_ENTRIES,
    ) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self.max_entries = max(0, int(max_entries))
        self._items: OrderedDict[tuple[int, int], PostingOccurrences] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()

    @staticmethod
    def _copy_value(value: PostingOccurrences) -> PostingOccurrences:
        # Copy row slices so one cached posting does not retain the entire
        # Arrow result table containing unrelated rows.
        return PostingOccurrences(
            positions=np.asarray(value.positions, dtype=np.int32).copy(),
            wrapper_flags=np.asarray(value.wrapper_flags, dtype=str).copy(),
        )

    @staticmethod
    def _value_bytes(value: PostingOccurrences) -> int:
        return int(value.positions.nbytes + value.wrapper_flags.nbytes)

    def get_many(
        self,
        keys: Sequence[tuple[int, int]],
    ) -> dict[tuple[int, int], PostingOccurrences]:
        found: dict[tuple[int, int], PostingOccurrences] = {}
        with self._lock:
            for raw_key in keys:
                key = (int(raw_key[0]), int(raw_key[1]))
                value = self._items.get(key)
                if value is None:
                    continue
                self._items.move_to_end(key)
                found[key] = value
        return found

    def put_many(
        self,
        values: Mapping[tuple[int, int], PostingOccurrences],
    ) -> dict[tuple[int, int], PostingOccurrences]:
        stored: dict[tuple[int, int], PostingOccurrences] = {}
        if not self.max_bytes or not self.max_entries:
            return stored

        with self._lock:
            for raw_key, raw_value in values.items():
                key = (int(raw_key[0]), int(raw_key[1]))
                value = self._copy_value(raw_value)
                value_bytes = self._value_bytes(value)
                if value_bytes > self.max_bytes:
                    continue

                previous = self._items.pop(key, None)
                if previous is not None:
                    self._bytes -= self._value_bytes(previous)

                self._items[key] = value
                self._bytes += value_bytes
                stored[key] = value

                while (
                    self._bytes > self.max_bytes
                    or len(self._items) > self.max_entries
                ):
                    _old_key, old_value = self._items.popitem(last=False)
                    self._bytes -= self._value_bytes(old_value)

                if key not in self._items:
                    stored.pop(key, None)

        return stored

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def stats(self) -> tuple[int, int]:
        with self._lock:
            return len(self._items), self._bytes


@dataclass(slots=True)
class MatchArrays:
    """Compact in-memory occurrence representation for one query term/doc."""

    starts: np.ndarray
    ends: np.ndarray
    term_ids: np.ndarray
    edit_distances: np.ndarray
    wrapper_flags: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=str)
    )
    # Optional per-occurrence detail used by composite phrase matches. Ordinary
    # single-token matches leave this empty and continue using wrapper_flags.
    component_annotations: list[list[dict]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Preserve compatibility with callers that still construct the
        # original four-array shape. Every occurrence receives an explicit
        # empty annotation in that case.
        if self.wrapper_flags.size == 0 and self.starts.size:
            self.wrapper_flags = np.full(self.starts.size, "", dtype=str)
        elif self.wrapper_flags.size != self.starts.size:
            raise ValueError(
                "MatchArrays.wrapper_flags must contain one entry per occurrence."
            )
        if self.component_annotations and len(self.component_annotations) != self.starts.size:
            raise ValueError(
                "MatchArrays.component_annotations must contain one entry per occurrence."
            )

    @classmethod
    def empty(cls) -> "MatchArrays":
        return cls(
            starts=np.empty(0, dtype=np.int32),
            ends=np.empty(0, dtype=np.int32),
            term_ids=np.empty(0, dtype=np.int32),
            edit_distances=np.empty(0, dtype=np.int32),
            wrapper_flags=np.empty(0, dtype=str),
        )

    def __len__(self) -> int:
        return int(self.starts.size)


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
    posting_cache: PostingLRUCache = field(default_factory=PostingLRUCache)
    # Lazily built only when a suffix-only wildcard query needs it. These
    # per-length sorted reversed terms allow suffix constraints to use bisect
    # without adding load-time cost or changing the persistent schema.
    normal_terms_by_reversed_length: dict[int, list[str]] = field(default_factory=dict)

# =============================================================================
# Tokenization and query normalization
# =============================================================================

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

_STANDARD_TOKEN_RE = re.compile(r"^\w+$", re.UNICODE)

ADDITIONAL_TOKEN_SPECIALS = set(
    '!"§$%&/()_=.:;#~><°-—+,'
    "'{}[]@„“|”’‘"
    "\\"
    "“”„«»〝〞〟＂❝❞❠❮❯‟«»"
)

_ADDITIONAL_TOKEN_SPECIALS_ESCAPED = re.escape("".join(ADDITIONAL_TOKEN_SPECIALS))

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
    "<": ">",
    "«": "»",
    "„": "“",
    "“": "”",
    "‘": "’",
    "‚": "’",
    "❞": "❝",
    "＂": "＂",
    "‹": "›",
    "〝": "〞",
    "″": "″",
    "〈": "〉",
    "｛": "｝",
    "（": "）",
    "❮": "❯",
    "❰": "❱",
    "❝": "❞",
    "⟨": "⟩",
    '"': '"',
    "'": "'",
    "`": "`",
    "′": "′",
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


_EXPLICIT_PATTERN_CHARS = set(r"*?^$+|{}[]\\")

_ADDITIONAL_TRAILING_PUNCTUATION = ",.;:!?"


def _strip_additional_wrapper(token: str) -> str:
    """Remove one balanced wrapper pair defined by the wrapper map."""
    if not _is_wrapped_additional_token(token):
        return token
    return token[1:-1]


def query_wrapper_flag(query_term: str) -> str | None:
    """Return an explicitly requested wrapper pair, or ``None`` if unrestricted.

    A wrapper is recognized only when it forms one complete configured pair
    around a non-empty token. This keeps ordinary unwrapped searches
    unrestricted while allowing queries such as ``\"hello\"`` and ``(hello)``
    to constrain individual occurrences.
    """
    query = normalize_token(query_term.strip())
    if not _is_wrapped_additional_token(query):
        return None
    return query[0] + query[-1]


def canonical_query_term(query_term: str) -> str:
    """Remove one explicitly supplied wrapper without changing token text."""
    query = query_term.strip()
    if query_wrapper_flag(query) is None:
        return query
    return query[1:-1]


def _phrase_boundary_character(char: str) -> bool:
    """Return True for literal punctuation/symbols outside an indexed token.

    ``*`` and ``?`` retain the search engine's established wildcard meaning,
    including when they occur at the end of a whitespace-delimited component.
    """
    if char in {"*", "?"}:
        return False
    category = unicodedata.category(char)
    return bool(category) and category[0] in {"P", "S"}


def _compile_phrase_boundary(text: str) -> tuple[re.Pattern[str] | None, int]:
    """Compile literal punctuation with each whitespace run matching one whitespace char.

    Phrase adjacency deliberately permits exactly one whitespace character for
    each whitespace boundary in the query. Punctuation and symbols are literal.
    The returned length is therefore the exact source span length to probe.
    """
    if not text:
        return None, 0
    parts: list[str] = []
    expected_length = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            while index < len(text) and text[index].isspace():
                index += 1
            parts.append(r"\s")
            expected_length += 1
            continue
        parts.append(re.escape(char))
        expected_length += 1
        index += 1
    try:
        return re.compile("".join(parts)), expected_length
    except re.error:  # pragma: no cover - escaped literal pieces are always valid
        return None, 0


def parse_phrase_query(query: str) -> PhraseSpec | None:
    """Parse a whitespace-containing query into tokens, wildcards and gaps.

    Whitespace-delimited chunks follow the same normal/additional term resolver
    as ordinary searches. Boundary punctuation is kept in the phrase's literal
    separators, rather than looked up as a separate vocabulary term.
    """
    value = query.strip()
    if not any(char.isspace() for char in value):
        return None

    components: list[PhraseComponent] = []
    for match in re.finditer(r"\S+", value):
        raw = match.group(0)
        left = 0
        right = len(raw)
        query_term = raw
        wildcard_pattern = raw if re.fullmatch(r"[*?]+", raw) else None

        # A standalone run of wildcard operators is a positional component,
        # not a vocabulary query. Embedded patterns such as ``plag*`` remain
        # ordinary indexed wildcard terms and are optimized by the resolver.
        if wildcard_pattern is None:
            route = classify_query(raw)
            # Keep an already-recognized normal/additional term intact
            # (important for internal punctuation, embedded wildcards, regexes
            # and wrapped tokens).
            if route == "unindexed":
                # If the entire chunk is not an indexed token, peel only literal
                # boundary punctuation/symbols. Those characters remain part
                # of the separator checked against the indexed source text.
                if "*" not in raw and "?" not in raw:
                    while left < right and _phrase_boundary_character(raw[left]):
                        left += 1
                    while right > left and _phrase_boundary_character(raw[right - 1]):
                        right -= 1
                    if left >= right:
                        continue
                    candidate = raw[left:right]
                    route = classify_query(candidate)
                    if route == "unindexed":
                        # Unknown punctuation-bearing chunks remain literal separators
                        # in the gap between the nearest indexed components.
                        continue
                    query_term = candidate

        wrapper = query_wrapper_flag(query_term) if wildcard_pattern is None else None
        component_start = match.start() + left
        component_end = match.start() + right
        if wrapper is not None:
            component_start += 1
            component_end -= 1
        if component_start >= component_end:
            continue
        components.append(
            PhraseComponent(
                query_term=query_term,
                query_start=int(component_start),
                query_end=int(component_end),
                required_wrapper=wrapper,
                wildcard_pattern=wildcard_pattern,
            )
        )

    # A phrase needs at least two components and at least one real indexed
    # anchor. A wildcard-only phrase uses the positional source verifier.
    if len(components) < 2 or not any(
        component.wildcard_pattern is None for component in components
    ):
        return None

    gap_patterns: list[re.Pattern[str]] = []
    gap_lengths: list[int] = []
    for left_component, right_component in zip(components, components[1:]):
        separator = value[left_component.query_end:right_component.query_start]
        pattern, length = _compile_phrase_boundary(separator)
        if pattern is None or length <= 0:
            return None
        gap_patterns.append(pattern)
        gap_lengths.append(length)

    prefix_pattern, prefix_length = _compile_phrase_boundary(
        value[:components[0].query_start]
    )
    suffix_pattern, suffix_length = _compile_phrase_boundary(
        value[components[-1].query_end:]
    )
    return PhraseSpec(
        components=tuple(components),
        gap_patterns=tuple(gap_patterns),
        gap_lengths=tuple(gap_lengths),
        prefix_pattern=prefix_pattern,
        prefix_length=prefix_length,
        suffix_pattern=suffix_pattern,
        suffix_length=suffix_length,
    )


def _phrase_edge_matches(
    content: str,
    position: int,
    pattern: re.Pattern[str] | None,
    length: int,
    *,
    before: bool,
) -> bool:
    if pattern is None:
        return True
    start, end = (position - length, position) if before else (position, position + length)
    if start < 0 or end > len(content):
        return False
    return pattern.fullmatch(content[start:end]) is not None


def _positional_wildcard_token_matches(pattern: str, token: str) -> bool:
    """Match a complete normal word against a standalone * / ? component.

    Every ``?`` consumes exactly one character and every ``*`` consumes at
    least one. The complete token is capped at 100 characters, matching the
    established wildcard length limit.
    """
    if len(token) > FTS_MAX_CHUNK_LENGTH or len(token) < len(pattern):
        return False
    if "*" not in pattern:
        return len(token) == len(pattern)
    return True


def _wildcard_word_starting_at(
    content: str,
    start: int,
    pattern: str,
) -> tuple[int, int] | None:
    """Return the normal-token span beginning at ``start`` if it fits pattern."""
    if start < 0 or start >= len(content):
        return None
    match = TOKEN_PATTERN.match(content, start)
    if match is None or match.start() != start:
        return None
    token = match.group(0)
    if not _positional_wildcard_token_matches(pattern, token):
        return None
    return int(match.start()), int(match.end())


def _wildcard_word_ending_at(
    content: str,
    end: int,
    pattern: str,
) -> tuple[int, int] | None:
    """Return the normal-token span ending at ``end`` without scanning a doc.

    At most ``FTS_MAX_CHUNK_LENGTH + 1`` characters are inspected. A longer
    token is rejected as soon as the configured wildcard bound is exceeded.
    """
    if end <= 0 or end > len(content):
        return None
    start = end
    while start > 0 and _WORD_CHAR_RE.fullmatch(content[start - 1]):
        start -= 1
        if end - start > FTS_MAX_CHUNK_LENGTH:
            return None
    if start == end:
        return None
    token = content[start:end]
    if not _positional_wildcard_token_matches(pattern, token):
        return None
    return int(start), int(end)


def phrase_position_candidate_docs(
    phrase: PhraseSpec,
    component_mappings: Sequence[dict[int, MatchArrays] | None],
    candidate_docs: set[int],
) -> set[int]:
    """Safely prefilter phrase documents using indexed token coordinates only.

    Each contiguous run of indexed phrase components must form an ordered
    chain whose token-end/token-start distances equal the fixed gap lengths.
    The actual punctuation/whitespace characters, phrase edges and standalone
    wildcard words are still checked against indexed source text later.
    """
    if len(component_mappings) != len(phrase.components):
        return set()

    indexed_indices = [
        index for index, component in enumerate(phrase.components)
        if component.wildcard_pattern is None
    ]
    if not indexed_indices:
        return set()

    possible_docs = set(candidate_docs)
    for component_index in indexed_indices:
        mapping = component_mappings[component_index]
        if mapping is None:
            return set()
        possible_docs.intersection_update(mapping)
        if not possible_docs:
            return set()

    # Build contiguous runs. A standalone positional wildcard breaks a run;
    # indexed components on either side can still be checked independently.
    indexed_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for component_index, component in enumerate(phrase.components):
        if component.wildcard_pattern is None:
            if run_start is None:
                run_start = component_index
        elif run_start is not None:
            if component_index - run_start > 1:
                indexed_runs.append((run_start, component_index - 1))
            run_start = None
    if run_start is not None and len(phrase.components) - run_start > 1:
        indexed_runs.append((run_start, len(phrase.components) - 1))

    if not indexed_runs:
        return possible_docs

    for doc_id in tuple(possible_docs):
        keep_document = True
        for run_start, run_end in indexed_runs:
            first_mapping = component_mappings[run_start]
            assert first_mapping is not None
            first_matches = first_mapping.get(doc_id)
            if first_matches is None or not len(first_matches):
                keep_document = False
                break

            # At each step retain only occurrences that can extend a valid
            # chain from the preceding component. This avoids false positives
            # caused by satisfying different phrase edges with unrelated
            # occurrences of a repeated middle word.
            possible_ends = first_matches.ends
            for left_index in range(run_start, run_end):
                right_mapping = component_mappings[left_index + 1]
                assert right_mapping is not None
                right_matches = right_mapping.get(doc_id)
                if right_matches is None or not len(right_matches):
                    keep_document = False
                    break

                order = np.argsort(right_matches.starts, kind="stable")
                right_starts = right_matches.starts[order]
                target_starts = possible_ends + int(phrase.gap_lengths[left_index])
                offsets = np.searchsorted(right_starts, target_starts, side="left")
                in_bounds = offsets < right_starts.size
                safe_offsets = np.minimum(offsets, right_starts.size - 1)
                matched_targets = np.unique(
                    target_starts[in_bounds & (right_starts[safe_offsets] == target_starts)]
                )
                if matched_targets.size == 0:
                    keep_document = False
                    break

                matched_occurrences = np.isin(right_starts, matched_targets)
                possible_ends = right_matches.ends[order[matched_occurrences]]
                if possible_ends.size == 0:
                    keep_document = False
                    break

            if not keep_document:
                break

        if not keep_document:
            possible_docs.discard(doc_id)

    return possible_docs


def build_ordered_phrase_matches(
    phrase: PhraseSpec,
    component_mappings: Sequence[dict[int, MatchArrays] | None],
    source_documents: dict[int, dict],
    candidate_docs: set[int],
) -> dict[int, MatchArrays]:
    """Combine indexed tokens and anchor-checked wildcard words into phrase hits.

    Standalone ``*`` / ``?`` components deliberately have no MatchArrays and
    are never expanded into vocabulary terms or every word occurrence in a
    document. The rarest non-wildcard component is selected as the anchor.
    Wildcard words are then checked directly at the exact source offset while
    extending left/right from that anchor.
    """
    if len(component_mappings) != len(phrase.components) or len(phrase.components) < 2:
        return {}

    indexed_component_indices = [
        i for i, component in enumerate(phrase.components)
        if component.wildcard_pattern is None
    ]
    if not indexed_component_indices:
        return {}

    phrase_docs = set(candidate_docs)
    for component_index in indexed_component_indices:
        mapping = component_mappings[component_index]
        if mapping is None:
            return {}
        phrase_docs.intersection_update(mapping)
        if not phrase_docs:
            return {}

    result: dict[int, MatchArrays] = {}
    for doc_id in sorted(phrase_docs):
        document = source_documents.get(int(doc_id))
        if document is None:
            continue
        content = str(document.get("content", ""))
        component_matches: list[MatchArrays | None] = []
        missing_component = False
        for component_index in range(len(phrase.components)):
            mapping = component_mappings[component_index]
            if mapping is None:
                component_matches.append(None)
                continue
            matches = mapping.get(int(doc_id))
            if matches is None or len(matches) == 0:
                missing_component = True
                break
            component_matches.append(matches)
        if missing_component:
            continue

        # Coordinate lookups use sorted NumPy arrays + searchsorted instead
        # of Python dictionaries. Starts and ends are prepared only for real
        # indexed components; wildcard positions are inspected lazily below.
        start_lookup_by_component: list[tuple[np.ndarray, np.ndarray] | None] = [
            None for _ in phrase.components
        ]
        end_lookup_by_component: list[tuple[np.ndarray, np.ndarray] | None] = [
            None for _ in phrase.components
        ]
        for component_index in indexed_component_indices:
            matches = component_matches[component_index]
            assert matches is not None
            start_order = np.argsort(matches.starts, kind="stable").astype(
                np.int32, copy=False
            )
            end_order = np.argsort(matches.ends, kind="stable").astype(
                np.int32, copy=False
            )
            start_lookup_by_component[component_index] = (
                matches.starts[start_order], start_order
            )
            end_lookup_by_component[component_index] = (
                matches.ends[end_order], end_order
            )

        def occurrences_at_coordinate(
            component_index: int,
            *,
            use_end: bool,
            coordinate: int,
        ) -> np.ndarray:
            lookup = (
                end_lookup_by_component[component_index]
                if use_end
                else start_lookup_by_component[component_index]
            )
            assert lookup is not None
            sorted_coordinates, order = lookup
            left = int(np.searchsorted(sorted_coordinates, coordinate, side="left"))
            right = int(np.searchsorted(sorted_coordinates, coordinate, side="right"))
            return order[left:right]

        anchor_component = min(
            indexed_component_indices,
            key=lambda index: (len(component_matches[index]), index),
        )
        # Indexed components store occurrence indices; positional wildcard
        # components store their one verified (start, end) pair.
        records: dict[tuple[int, ...], tuple[int, int, int, list[dict]]] = {}

        def occurrence_span(
            component_index: int,
            state: dict[int, int | tuple[int, int]],
        ) -> tuple[int, int]:
            value = state[component_index]
            if phrase.components[component_index].wildcard_pattern is not None:
                return value  # type: ignore[return-value]
            matches = component_matches[component_index]
            assert matches is not None
            occurrence_index = int(value)  # type: ignore[arg-type]
            return (
                int(matches.starts[occurrence_index]),
                int(matches.ends[occurrence_index]),
            )

        for anchor_occurrence in range(len(component_matches[anchor_component])):  # type: ignore[arg-type]
            states: list[dict[int, int | tuple[int, int]]] = [
                {anchor_component: anchor_occurrence}
            ]

            # Extend left from the anchor. For a wildcard predecessor, derive
            # its token end from the next component and inspect only that word.
            for component_index in range(anchor_component - 1, -1, -1):
                extended: list[dict[int, int | tuple[int, int]]] = []
                gap_length = phrase.gap_lengths[component_index]
                gap_pattern = phrase.gap_patterns[component_index]
                for state in states:
                    next_start, _next_end = occurrence_span(component_index + 1, state)
                    target_end = next_start - gap_length
                    if gap_pattern.fullmatch(content[target_end:next_start]) is None:
                        continue

                    wildcard_pattern = phrase.components[component_index].wildcard_pattern
                    if wildcard_pattern is not None:
                        span = _wildcard_word_ending_at(
                            content,
                            target_end,
                            wildcard_pattern,
                        )
                        if span is None:
                            continue
                        new_state = dict(state)
                        new_state[component_index] = span
                        extended.append(new_state)
                        continue

                    current_matches = component_matches[component_index]
                    assert current_matches is not None
                    for previous_occurrence in occurrences_at_coordinate(
                        component_index,
                        use_end=True,
                        coordinate=target_end,
                    ):
                        previous_end = int(current_matches.ends[previous_occurrence])
                        if gap_pattern.fullmatch(content[previous_end:next_start]) is None:
                            continue
                        new_state = dict(state)
                        new_state[component_index] = previous_occurrence
                        extended.append(new_state)
                states = extended
                if not states:
                    break

            if not states:
                continue

            # Extend right. Wildcard components are checked at their one
            # expected start offset; indexed components use their sorted map.
            for component_index in range(anchor_component, len(phrase.components) - 1):
                extended = []
                gap_length = phrase.gap_lengths[component_index]
                gap_pattern = phrase.gap_patterns[component_index]
                wildcard_pattern = phrase.components[component_index + 1].wildcard_pattern
                for state in states:
                    current_start, current_end = occurrence_span(component_index, state)
                    target_start = current_end + gap_length
                    if gap_pattern.fullmatch(content[current_end:target_start]) is None:
                        continue

                    if wildcard_pattern is not None:
                        span = _wildcard_word_starting_at(
                            content,
                            target_start,
                            wildcard_pattern,
                        )
                        if span is None:
                            continue
                        new_state = dict(state)
                        new_state[component_index + 1] = span
                        extended.append(new_state)
                        continue

                    next_matches = component_matches[component_index + 1]
                    assert next_matches is not None
                    for next_occurrence in occurrences_at_coordinate(
                        component_index + 1,
                        use_end=False,
                        coordinate=target_start,
                    ):
                        next_start = int(next_matches.starts[next_occurrence])
                        if gap_pattern.fullmatch(content[current_end:next_start]) is None:
                            continue
                        new_state = dict(state)
                        new_state[component_index + 1] = next_occurrence
                        extended.append(new_state)
                states = extended
                if not states:
                    break

            for state in states:
                if len(state) != len(phrase.components):
                    continue
                first_start, _first_end = occurrence_span(0, state)
                _last_start, last_end = occurrence_span(len(phrase.components) - 1, state)
                if not _phrase_edge_matches(
                    content,
                    first_start,
                    phrase.prefix_pattern,
                    phrase.prefix_length,
                    before=True,
                ):
                    continue
                if not _phrase_edge_matches(
                    content,
                    last_end,
                    phrase.suffix_pattern,
                    phrase.suffix_length,
                    before=False,
                ):
                    continue

                component_annotations: list[dict] = []
                total_edit_distance = 0
                for component_index, component in enumerate(phrase.components):
                    start, end = occurrence_span(component_index, state)
                    matches = component_matches[component_index]
                    if component.wildcard_pattern is not None:
                        term_id = FALLBACK_TERM_ID
                        edit_distance = 0
                        wrapper_flag = wrapper_flag_for_span(content, start, end)
                    else:
                        assert matches is not None
                        value = int(state[component_index])  # type: ignore[arg-type]
                        term_id = int(matches.term_ids[value])
                        edit_distance = int(matches.edit_distances[value])
                        wrapper_flag = str(matches.wrapper_flags[value])
                    total_edit_distance += edit_distance
                    component_annotations.append(
                        {
                            "query_term": component.query_term,
                            "start": start,
                            "end": end,
                            "wrapper_flag": wrapper_flag,
                            "term_id": term_id,
                            "edit_distance": edit_distance,
                        }
                    )

                match_start = first_start - phrase.prefix_length
                match_end = last_end + phrase.suffix_length
                signature = tuple(
                    value
                    for component_index in range(len(phrase.components))
                    for value in occurrence_span(component_index, state)
                )
                existing = records.get(signature)
                if existing is None or total_edit_distance < existing[2]:
                    records[signature] = (
                        match_start,
                        match_end,
                        total_edit_distance,
                        component_annotations,
                    )

        if not records:
            continue

        ordered_records = sorted(
            records.values(),
            key=lambda item: (item[0], item[1], item[2]),
        )
        result[int(doc_id)] = MatchArrays(
            starts=np.asarray([item[0] for item in ordered_records], dtype=np.int32),
            ends=np.asarray([item[1] for item in ordered_records], dtype=np.int32),
            term_ids=np.full(len(ordered_records), FALLBACK_TERM_ID, dtype=np.int32),
            edit_distances=np.asarray([item[2] for item in ordered_records], dtype=np.int32),
            wrapper_flags=np.full(len(ordered_records), "", dtype=str),
            component_annotations=[item[3] for item in ordered_records],
        )

    return result


def canonicalize_additional_token(
    token: str,
    *,
    strip_trailing_punctuation: bool = True,
) -> str:
    """Return the canonical additional term using the configured wrapper pairs.

    Source tokens strip ordinary prose-closing punctuation first. Query terms
    keep such punctuation so forms like ``M./Hormel,`` and ``(1902/2007):``
    remain unindexed in database mode. Balanced wrappers themselves are canonicalized
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
        return "unindexed"

    # Ordinary closing punctuation is not an indexed query. A trailing
    # question mark is the documented one-character wildcard when the complete
    # query otherwise has valid wildcard syntax. A complete configured wrapper
    # pair takes precedence over regex metacharacters such as [] and {}.
    # Wrapped plain words route through the normal-term index (their wrapper is
    # recorded per occurrence in postings); other wrapped tokens use additional.
    if query.rstrip(_ADDITIONAL_TRAILING_PUNCTUATION) != query:
        trailing_question_wildcard = (
            query.endswith("?")
            and re.fullmatch(r"[\w*?^$+|{}\[\]\\.\-]+", query, flags=re.UNICODE)
        )
        if not trailing_question_wildcard:
            return "unindexed"
    if _is_wrapped_additional_token(query):
        if _STANDARD_TOKEN_RE.fullmatch(query[1:-1]):
            return "normal"
        return "additional"

    canonical = canonicalize_additional_token(
        query,
        strip_trailing_punctuation=False,
    )
    if canonical != query and is_indexable_additional_token(canonical):
        return "additional"
    if _STANDARD_TOKEN_RE.fullmatch(query):
        return "normal"
    # Explicit regex/wildcard operators must take precedence over the
    # additional-token route. Otherwise patterns such as ``foo*bar`` and
    # ``c?t`` are misclassified as literal punctuation-bearing tokens simply
    # because their first and last characters are word characters.
    if any(char in _EXPLICIT_PATTERN_CHARS for char in query):
        if re.fullmatch(r"[\w*?^$+|{}\[\]\\.\-]+", query, flags=re.UNICODE):
            return "normal"
    if is_indexable_additional_token(query):
        return "additional"
    return "unindexed"


def _additional_token_core(raw_token: str) -> str:
    """Return canonical source-token text after prose punctuation and wrappers."""
    return canonicalize_additional_token(
        raw_token,
        strip_trailing_punctuation=True,
    )


def normalize_case_flags(case_sensitive: Sequence[bool] | None, count: int) -> list[bool]:
    if case_sensitive is None:
        return [False] * count
    flags = [bool(value) for value in case_sensitive]
    if len(flags) < count:
        flags.extend([False] * (count - len(flags)))
    return flags[:count]


def normalize_term_modes(
    term_modes: Sequence[TermSearchMode] | None,
    count: int,
    default_mode: TermSearchMode = "regex",
) -> list[TermSearchMode]:
    """Return one validated search mode for each query-term slot."""
    valid_modes = {"regex", "fuzzy", "both"}
    if term_modes is None:
        values: list[str] = [default_mode] * count
    else:
        values = [str(value).strip().casefold() for value in term_modes]
        if len(values) < count:
            values.extend([default_mode] * (count - len(values)))
        values = values[:count]
    invalid = [value for value in values if value not in valid_modes]
    if invalid:
        raise ValueError(
            "Each term mode must be 'regex', 'fuzzy', or 'both'; "
            f"invalid value(s): {invalid!r}."
        )
    return values  # type: ignore[return-value]


def combined_term_mode(term_modes: Sequence[TermSearchMode]) -> str:
    """Summarize a term-mode list for reports and document headers."""
    if not term_modes:
        return "regex"
    first = term_modes[0]
    return first if all(value == first for value in term_modes) else "mixed"


_GERMAN_SAME_LENGTH_UPPER_LOWER_UTF8 = (
    ("Ä".encode("utf-8"), "ä".encode("utf-8")),
    ("Ö".encode("utf-8"), "ö".encode("utf-8")),
    ("Ü".encode("utf-8"), "ü".encode("utf-8")),
)
_GERMAN_UPPER_LOWER_UTF8 = (
    *_GERMAN_SAME_LENGTH_UPPER_LOWER_UTF8,
    ("ẞ".encode("utf-8"), "ß".encode("utf-8")),
)
_GERMAN_CAPITAL_SHARP_S_UTF8 = "ẞ".encode("utf-8")
_GERMAN_SMALL_SHARP_S_UTF8 = "ß".encode("utf-8")


def literal_english_german_lower_bytes_with_offsets(
    value: bytes,
) -> tuple[bytes, tuple[int, ...]]:
    """Fold English/German case pairs in bytes and record shortened ẞ offsets.

    ASCII lowercasing and Ä/Ö/Ü conversion preserve byte lengths. Capital ẞ
    (three UTF-8 bytes) becomes ß (two bytes), so its normalized positions are
    returned for restoring original source-byte offsets after matching.
    """
    lowered = value.lower()
    for uppercase, lowercase in _GERMAN_SAME_LENGTH_UPPER_LOWER_UTF8:
        lowered = lowered.replace(uppercase, lowercase)

    capital_sharp_s_positions: list[int] = []
    search_from = 0
    while True:
        found_at = lowered.find(_GERMAN_CAPITAL_SHARP_S_UTF8, search_from)
        if found_at < 0:
            break
        capital_sharp_s_positions.append(found_at)
        search_from = found_at + len(_GERMAN_CAPITAL_SHARP_S_UTF8)

    # Replacing each ẞ shortens the byte string by one byte. Store its
    # position in the resulting normalized string, not the original buffer.
    normalized_positions = tuple(
        position - index
        for index, position in enumerate(capital_sharp_s_positions)
    )
    if capital_sharp_s_positions:
        lowered = lowered.replace(
            _GERMAN_CAPITAL_SHARP_S_UTF8,
            _GERMAN_SMALL_SHARP_S_UTF8,
        )
    return lowered, normalized_positions


def literal_english_german_lower_bytes(value: bytes) -> bytes:
    """Return case-folded English/German bytes without decoding text."""
    return literal_english_german_lower_bytes_with_offsets(value)[0]


def restore_literal_byte_match_offsets(
    matches: MatchArrays,
    shortened_character_positions: tuple[int, ...],
) -> None:
    """Map normalized byte offsets back to source offsets after ẞ -> ß."""
    if not shortened_character_positions or len(matches) == 0:
        return
    shortened_positions = np.asarray(
        shortened_character_positions,
        dtype=np.int32,
    )
    start_adjustments = np.searchsorted(
        shortened_positions, matches.starts, side="left"
    ).astype(np.int32, copy=False)
    end_adjustments = np.searchsorted(
        shortened_positions, matches.ends, side="left"
    ).astype(np.int32, copy=False)
    matches.starts = matches.starts + start_adjustments
    matches.ends = matches.ends + end_adjustments


def literal_query_needs_case_folding(encoded_query: bytes) -> bool:
    """Return whether a query contains English or German cased letters."""
    if any(65 <= value <= 90 or 97 <= value <= 122 for value in encoded_query):
        return True
    return any(
        uppercase in encoded_query or lowercase in encoded_query
        for uppercase, lowercase in _GERMAN_UPPER_LOWER_UTF8
    )


def find_literal_byte_matches(haystack: bytes, needle: bytes) -> MatchArrays:
    """Find every overlapping exact UTF-8 byte-substring occurrence."""
    if not haystack or not needle or len(needle) > len(haystack):
        return MatchArrays.empty()

    starts: list[int] = []
    search_from = 0
    while True:
        found_at = haystack.find(needle, search_from)
        if found_at < 0:
            break
        starts.append(found_at)
        # Advance by one byte to preserve overlapping occurrences. Encoded
        # UTF-8 needles can only begin at a valid character boundary.
        search_from = found_at + 1

    if not starts:
        return MatchArrays.empty()

    starts_array = np.asarray(starts, dtype=np.int32)
    return MatchArrays(
        starts=starts_array,
        ends=starts_array + np.int32(len(needle)),
        term_ids=np.full(len(starts), FALLBACK_TERM_ID, dtype=np.int32),
        edit_distances=np.zeros(len(starts), dtype=np.int32),
    )


def decode_literal_content_and_convert_byte_positions(
    raw_content: bytes,
    match_arrays: Sequence[tuple[MatchArrays, bool]],
) -> str:
    """Decode one raw candidate and map UTF-8 byte matches to char offsets.

    The document is decoded in sorted chunks ending at requested match
    boundaries. This produces the full Unicode string and byte-to-character
    positions in one pass, avoiding a second full decode or a per-byte map.
    """
    requested_offsets: set[int] = set()
    for matches, uses_byte_offsets in match_arrays:
        if not uses_byte_offsets or matches.starts.size == 0:
            continue
        requested_offsets.update(int(value) for value in matches.starts)
        requested_offsets.update(int(value) for value in matches.ends)

    if not requested_offsets:
        return raw_content.decode("utf-8", errors="ignore")

    byte_to_character: dict[int, int] = {}
    decoded_parts: list[str] = []
    previous_byte = 0
    character_position = 0

    for byte_offset in sorted(requested_offsets):
        segment = raw_content[previous_byte:byte_offset].decode(
            "utf-8", errors="ignore"
        )
        decoded_parts.append(segment)
        character_position += len(segment)
        byte_to_character[byte_offset] = character_position
        previous_byte = byte_offset

    decoded_parts.append(
        raw_content[previous_byte:].decode("utf-8", errors="ignore")
    )
    content = "".join(decoded_parts)

    for matches, uses_byte_offsets in match_arrays:
        if not uses_byte_offsets or matches.starts.size == 0:
            continue
        matches.starts = np.fromiter(
            (byte_to_character[int(value)] for value in matches.starts),
            dtype=np.int32,
            count=matches.starts.size,
        )
        matches.ends = np.fromiter(
            (byte_to_character[int(value)] for value in matches.ends),
            dtype=np.int32,
            count=matches.ends.size,
        )

    return content


def _case_sensitive_regex_match(query_term: str, actual: str) -> bool:
    """Apply the engine's regex/wildcard semantics with case preserved."""
    query = query_term.strip()
    if "*" in query or "?" in query:
        pattern = compile_wildcard(query, case_sensitive=True)
        return pattern is not None and pattern.fullmatch(actual) is not None
    if _has_regex_syntax_except_wildcards(query):
        try:
            return re.fullmatch(query, actual) is not None
        except re.error:
            return False
    return actual == query


def filter_case_sensitive_matches(
    matches: "MatchArrays",
    content: str,
    query_term: str,
    mode: TermSearchMode,
    fuzzy_distance: int,
) -> "MatchArrays":
    """Filter normalized-index hits against source spelling/case.

    For ``both``, a candidate survives when either the case-sensitive regex /
    wildcard expression matches or its original text is within the fuzzy edit
    distance. Regex matches are assigned edit distance zero.
    """
    if len(matches) == 0:
        return matches

    kept_starts: list[int] = []
    kept_ends: list[int] = []
    kept_term_ids: list[int] = []
    kept_distances: list[int] = []
    kept_wrapper_flags: list[str] = []
    for start, end, term_id, edit_distance, wrapper_flag in zip(
        matches.starts.tolist(),
        matches.ends.tolist(),
        matches.term_ids.tolist(),
        matches.edit_distances.tolist(),
        matches.wrapper_flags.tolist(),
    ):
        actual = content[int(start):int(end)]
        regex_match = (
            _case_sensitive_regex_match(query_term, actual)
            if mode in {"regex", "both"}
            else False
        )
        fuzzy_edit = (
            int(Levenshtein.distance(query_term.strip(), actual))
            if mode in {"fuzzy", "both"}
            else None
        )
        fuzzy_match = fuzzy_edit is not None and fuzzy_edit <= fuzzy_distance

        if mode == "regex":
            if not regex_match:
                continue
            edit = 0
        elif mode == "fuzzy":
            if not fuzzy_match:
                continue
            edit = int(fuzzy_edit or 0)
        else:  # both
            if not regex_match and not fuzzy_match:
                continue
            edit = 0 if regex_match else int(fuzzy_edit or 0)

        kept_starts.append(int(start))
        kept_ends.append(int(end))
        kept_term_ids.append(int(term_id))
        kept_distances.append(int(edit))
        kept_wrapper_flags.append(str(wrapper_flag))

    if not kept_starts:
        return MatchArrays.empty()
    return MatchArrays(
        starts=np.asarray(kept_starts, dtype=np.int32),
        ends=np.asarray(kept_ends, dtype=np.int32),
        term_ids=np.asarray(kept_term_ids, dtype=np.int32),
        edit_distances=np.asarray(kept_distances, dtype=np.int32),
        wrapper_flags=np.asarray(kept_wrapper_flags, dtype=str),
    )


def normalize_token(token: str) -> str:
    return token.casefold()


def iter_tokens(content: str):
    for match in TOKEN_PATTERN.finditer(content):
        yield normalize_token(match.group(0)), match.start(), match.end()


def wrapper_flag_for_span(content: str, start: int, end: int) -> str:
    """Return the balanced wrapper pair directly surrounding one plain-word span.

    ``""`` means the occurrence is not wrapped; otherwise a two-character pair
    such as ``"()"``, ``"[]"`` or ``"«»"`` records which wrappers surrounded it.
    """
    if start == 0 or end >= len(content):
        return ""
    opening = content[start - 1]
    closing = _ADDITIONAL_TOKEN_WRAPPER_PAIRS.get(opening)
    if closing is not None and content[end] == closing:
        return opening + closing
    return ""


def count_words(content: str) -> int:
    """Count words using the same tokenizer as the search index."""

    return sum(1 for _ in TOKEN_PATTERN.finditer(content))

# =============================================================================
# Source file access and folder scanning
# =============================================================================

def _read_file_worker(path_str: str) -> dict | None:
    """Read one source file once and calculate its SHA-256 from those bytes."""
    path = Path(path_str).absolute()
    try:
        raw_bytes = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_bytes": raw_bytes,
        }
    except Exception:
        logger.exception("Worker failed to read %s", path)
        return None


def _iter_read_file_workers(
    paths: Sequence[Path],
    *,
    max_workers: int = 16,
):
    """Yield completed file reads so duplicate byte buffers can be released early."""
    source_paths = collect_paths(paths)
    if not source_paths:
        return

    cpu_count = os.cpu_count() or 1
    worker_count = min(max(1, int(max_workers)), cpu_count, len(source_paths))
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="text-read",
    ) as executor:
        futures = {
            executor.submit(_read_file_worker, str(path)): str(path)
            for path in source_paths
        }
        for future in concurrent.futures.as_completed(futures):
            path_str = futures.pop(future)
            try:
                result = future.result()
            except Exception:
                logger.exception("Unexpected read-worker failure for %s", path_str)
                result = None
            yield path_str, result


def _append_grouped_occurrence(
    grouped_terms: dict[TermKey, dict[str, object]],
    term_key: TermKey,
    start: int,
    length: int,
    wrapper_flag: str,
) -> None:
    """Append one occurrence to a compact per-term integer array."""
    posting = grouped_terms.get(term_key)
    if posting is None:
        posting = {
            "positions": array("i"),
            "wrapper_flags": [],
        }
        grouped_terms[term_key] = posting

    positions = posting["positions"]
    if not isinstance(positions, array):
        raise TypeError("Internal posting positions buffer has an unexpected type")
    positions.extend((int(start), int(length)))
    posting["wrapper_flags"].append(wrapper_flag)


def _tokenize_content_worker(record: dict) -> dict | None:
    """Tokenize one already-read unique content buffer into grouped postings.

    ``word_count`` is collected during the normal-token pass. Position pairs are
    accumulated in compact C integer arrays and exposed as NumPy int32 views only
    after the token has been fully grouped.
    """
    path_str = str(record["path"])
    sha256 = str(record["sha256"])
    include_content = bool(record.get("include_content", True))
    try:
        raw_bytes = record.pop("raw_bytes")
        content = raw_bytes.decode("utf-8", errors="ignore")
        del raw_bytes
        grouped_terms: dict[TermKey, dict[str, object]] = {}
        word_count = 0

        # Normal terms and word count share the same regex traversal.
        for match in TOKEN_PATTERN.finditer(content):
            word_count += 1
            start = int(match.start())
            end = int(match.end())
            token = normalize_token(match.group(0))
            _append_grouped_occurrence(
                grouped_terms,
                ("normal", token),
                start,
                end - start,
                wrapper_flag_for_span(content, start, end),
            )

        # Additional terms have different semantics, so they require their own
        # scan. Avoid the old iter_additional_tokens() normalization result,
        # which was calculated and then discarded by the index builder.
        for match in _NON_WHITESPACE_TOKEN_RE.finditer(content):
            start = int(match.start())
            end = int(match.end())
            raw_token = content[start:end]
            raw_core = raw_token.rstrip(_ADDITIONAL_TRAILING_PUNCTUATION)
            if not raw_core:
                continue

            canonical = canonicalize_additional_token(raw_core)
            if not canonical or len(canonical) > FTS_MAX_CHUNK_LENGTH:
                continue
            wrapped = _is_wrapped_additional_token(raw_core)
            if not is_indexable_additional_token(canonical):
                continue

            span_start = start + (1 if wrapped else 0)
            wrapper_flag = raw_core[0] + raw_core[-1] if wrapped else ""
            _append_grouped_occurrence(
                grouped_terms,
                ("additional", canonical),
                span_start,
                len(canonical),
                wrapper_flag,
            )

        # The database expects flat [start, length, ...] int32 positions.
        # array('i') is a compact 32-bit signed integer buffer on supported
        # Windows/Linux builds; keep a safe conversion fallback for other ABIs.
        for posting in grouped_terms.values():
            position_buffer = posting["positions"]
            if not isinstance(position_buffer, array):
                raise TypeError("Internal posting positions buffer has an unexpected type")
            if position_buffer.itemsize == np.dtype(np.int32).itemsize:
                posting["positions"] = np.frombuffer(position_buffer, dtype=np.int32)
            else:
                posting["positions"] = np.asarray(position_buffer, dtype=np.int32)

        return {
            "path": path_str,
            "sha256": sha256,
            "word_count": int(word_count),
            "grouped_terms": grouped_terms,
            # Benchmark workers can omit the decoded source text to avoid
            # measuring large result-transfer costs when only tokenization is
            # under test. Production indexing keeps the default (True).
            "content": content if include_content else None,
        }
    except Exception:
        logger.exception("Worker failed to tokenize %s", path_str)
        return None


def _run_tokenize_content_workers(
    records: Sequence[dict],
    *,
    max_workers: int = TOKENIZATION_MAX_WORKERS,
    batch_size: int = TOKENIZATION_BATCH_SIZE,
    backend: Literal["processes", "threads"] = TOKENIZATION_BACKEND,
) -> list[dict | None]:
    """Tokenize unique contents using the measured process-pool defaults.

    The worker count never exceeds psutil's physical-core count. ``batch_size``
    maps directly to ``ProcessPoolExecutor.map(chunksize=...)``. A value of 1 was
    the fastest measured setting for the user's file collection. Threads remain
    available as an explicit fallback and use the same physical-core ceiling.
    """
    if not records:
        return []

    worker_count = get_tokenization_worker_count(
        task_count=len(records),
        requested_workers=max_workers,
    )

    if backend == "processes":
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
        ) as executor:
            return list(
                executor.map(
                    _tokenize_content_worker,
                    records,
                    chunksize=max(1, int(batch_size)),
                )
            )

    if backend == "threads":
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="text-tokenize",
        ) as executor:
            return list(executor.map(_tokenize_content_worker, records))

    raise ValueError(f"Unsupported tokenization backend: {backend!r}")

def read_text_with_sha256(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="ignore"), hashlib.sha256(raw).hexdigest()


def document_row(
    doc_id: int,
    path: Path,
    content: str,
    sha256: str,
    *,
    word_count: int | None = None,
) -> dict:
    """Build a document row, reusing a tokenizer-supplied word count when present."""
    return {
        "id": int(doc_id),
        "path": str(path.absolute()),
        "sha256": sha256,
        "content": content,
        "word_count": int(count_words(content) if word_count is None else word_count),
    }


def collect_paths(paths: Sequence[Path]) -> list[Path]:
    return sorted(
        {path.absolute() for path in paths},
        key=lambda path: str(path).casefold(),
    )


def normalized_path_key(path: str | Path) -> str:
    """Return a stable, platform-aware key for matching indexed file paths."""
    return os.path.normcase(os.path.normpath(os.path.abspath(os.fspath(path))))


def scan_search_folder(
    folder: Path,
    excluded_folder: Path,
) -> tuple[list[Path], dict[str, tuple[int, int, int, int, int]]]:
    """Enumerate .txt files and capture persistent change-detection fingerprints.

    A traversal/stat error aborts the scan. Callers must never interpret an
    incomplete scan as evidence that indexed documents were deleted.
    """
    folder = folder.absolute()
    excluded_folder = excluded_folder.absolute()
    if not folder.is_dir():
        raise OSError(f"Search folder is not accessible: {folder}")

    paths: list[Path] = []
    snapshot: dict[str, tuple[int, int, int, int, int]] = {}

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
            if not stat_module.S_ISREG(stat_result.st_mode):
                continue
            paths.append(path)
            snapshot[str(path)] = (
                int(stat_result.st_mtime_ns),
                int(stat_result.st_size),
                int(stat_result.st_ctime_ns),
                int(stat_result.st_dev),
                int(stat_result.st_ino),
            )

    return collect_paths(paths), snapshot


def path_in_folder(path: Path, folder: Path) -> bool:
    """Return True when path is inside folder (including folder itself)."""
    try:
        path.relative_to(folder)
        return True
    except ValueError:
        return False

# =============================================================================
# LanceDB persistence primitives
# =============================================================================

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
) -> tuple[pa.Table, bool, float, float]:
    """Execute FTS while measuring query-builder setup and Arrow materialization."""
    setup_started = time.perf_counter()
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
    setup_ms = (time.perf_counter() - setup_started) * 1000.0
    execution_ms = 0.0

    if supports_flag:
        started = time.perf_counter()
        try:
            arrow = query.to_arrow(disable_scoring_autoprojection=True)
            execution_ms += (time.perf_counter() - started) * 1000.0
            return arrow, False, setup_ms, execution_ms
        except TypeError:
            execution_ms += (time.perf_counter() - started) * 1000.0

    # Compatibility fallback for SDK versions where the scanner option is not
    # exposed by the high-level query object. The score is projected only as a
    # fallback and removed immediately after materialization.
    setup_started = time.perf_counter()
    score_query = query.select(["doc_id", "_score"])
    setup_ms += (time.perf_counter() - setup_started) * 1000.0
    started = time.perf_counter()
    arrow = score_query.to_arrow()
    execution_ms += (time.perf_counter() - started) * 1000.0
    if "_score" in getattr(arrow, "column_names", []):
        arrow = arrow.drop(["_score"])
    return arrow, True, setup_ms, execution_ms


def _record_predicate_construction(elapsed_ms: float) -> None:
    report = _ACTIVE_DB_QUERY_REPORT.get()
    if report is not None:
        report.db_predicate_construction_ms += float(elapsed_ms)


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


def _indexed_scalar_columns(table: object) -> set[str]:
    """Return scalar-indexed column names when the LanceDB API exposes them."""
    for method_name in ("list_indices", "list_indexes"):
        method = getattr(table, method_name, None)
        if not callable(method):
            continue
        try:
            entries = method()
        except Exception:
            continue

        columns: set[str] = set()
        try:
            for entry in entries or ():
                if isinstance(entry, Mapping):
                    raw = entry.get(
                        "columns",
                        entry.get("column", entry.get("column_name")),
                    )
                else:
                    raw = getattr(
                        entry,
                        "columns",
                        getattr(entry, "column", getattr(entry, "column_name", None)),
                    )
                if isinstance(raw, str):
                    columns.add(raw)
                elif raw:
                    try:
                        columns.update(str(value) for value in raw)
                    except TypeError:
                        pass
        except TypeError:
            continue
        return columns
    return set()


def _ensure_btree_scalar_index(
    table: object,
    column: str,
    *,
    index_name: str,
) -> bool:
    """Create a BTREE scalar index unless the table already has one."""
    if column in _indexed_scalar_columns(table):
        return True

    creator = getattr(table, "create_scalar_index", None)
    if not callable(creator):
        logger.warning(
            "LanceDB table %s does not support create_scalar_index(); "
            "queries will use the existing table layout",
            _table_cache_key(table),
        )
        return False

    kwargs: dict[str, object] = {}
    try:
        parameters = inspect.signature(creator).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if "index_type" in parameters or accepts_kwargs:
        kwargs["index_type"] = "BTREE"
    if "name" in parameters:
        kwargs["name"] = index_name
    if "replace" in parameters:
        kwargs["replace"] = False

    try:
        creator(column, **kwargs)
        logger.info(
            "Created LanceDB BTREE scalar index %s on %s.%s",
            index_name,
            _table_cache_key(table),
            column,
        )
        return True
    except Exception as exc:
        message = str(exc).casefold()
        if "already exists" in message or "index exists" in message:
            return True
        # Older bindings may expose an incomplete signature. Retry with the
        # simplest public call before treating index creation as unsupported.
        try:
            creator(column)
            logger.info(
                "Created LanceDB scalar index on %s.%s",
                _table_cache_key(table),
                column,
            )
            return True
        except Exception as retry_exc:
            retry_message = str(retry_exc).casefold()
            if "already exists" in retry_message or "index exists" in retry_message:
                return True
            logger.warning(
                "Could not create scalar index on %s.%s: %s",
                _table_cache_key(table),
                column,
                retry_exc,
            )
            return False


def ensure_search_scalar_indexes(
    postings_table: object,
    frequencies_table: object,
) -> dict[str, bool]:
    """Ensure indexes supporting the two hottest integer-filtered tables."""
    tables = {
        POSTINGS_TABLE: postings_table,
        TERM_FREQUENCIES_TABLE: frequencies_table,
    }
    result: dict[str, bool] = {}
    for table_name, columns in SEARCH_SCALAR_INDEXES.items():
        table = tables[table_name]
        for column in columns:
            key = f"{table_name}.{column}"
            result[key] = _ensure_btree_scalar_index(
                table,
                column,
                index_name=f"idx_{table_name}_{column}",
            )
    return result


def _record_arrow_query_result(
    table: object,
    query: object,
    arrow: pa.Table,
    *,
    operation: str,
    setup_ms: float,
    execution_ms: float,
    to_pylist_ms: float = 0.0,
    rows_returned: int | None = None,
) -> None:
    """Record every materialized LanceDB Arrow query, including raw-Arrow paths."""
    report = _ACTIVE_DB_QUERY_REPORT.get()
    if report is None:
        return

    row_count = int(
        getattr(arrow, "num_rows", 0) if rows_returned is None else rows_returned
    )
    arrow_bytes = int(getattr(arrow, "nbytes", 0) or 0)
    table_key = _table_cache_key(table)
    cache_state = "warm" if table_key in _DB_CACHE_SEEN_TABLES else "cold"
    _DB_CACHE_SEEN_TABLES.add(table_key)
    fragment_file_count = _query_fragment_file_count(query, arrow, table)

    report.db_query_setup_ms += float(setup_ms)
    report.db_execution_ms += float(execution_ms)
    report.db_to_pylist_ms += float(to_pylist_ms)
    # Inclusive: PyArrow creates Python row objects inside to_pylist(). Raw
    # Arrow consumers correctly contribute zero to this conversion metric.
    report.db_python_object_creation_ms += float(to_pylist_ms)
    report.db_queries += 1
    report.db_rows_returned += row_count
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
        "setup_ms": float(setup_ms),
        "execution_ms": float(execution_ms),
        "to_pylist_ms": float(to_pylist_ms),
        "python_object_creation_ms": float(to_pylist_ms),
        "rows": row_count,
        "bytes": arrow_bytes,
        "fragment_files_touched": fragment_file_count,
    })


def _record_db_metadata_call(operation: str, elapsed_ms: float) -> None:
    """Record LanceDB metadata calls that do not materialize an Arrow result."""
    report = _ACTIVE_DB_QUERY_REPORT.get()
    if report is None:
        return
    report.db_metadata_ms += float(elapsed_ms)
    report.db_metadata_calls += 1
    report.db_metadata_breakdown.append({
        "operation": operation,
        "time_ms": float(elapsed_ms),
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

    started = time.perf_counter()
    rows = arrow.to_pylist()
    to_pylist_ms = (time.perf_counter() - started) * 1000.0

    _record_arrow_query_result(
        table,
        query,
        arrow,
        operation=operation,
        setup_ms=setup_ms,
        execution_ms=execution_ms,
        to_pylist_ms=to_pylist_ms,
        rows_returned=len(rows),
    )
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


def query_arrow_columns(
    table: object,
    predicate: str | None = None,
    columns: list[str] | None = None,
    *,
    operation: str = "LanceDB Arrow query",
) -> pa.Table:
    """Fetch raw Arrow columns and instrument the query without Python-row conversion."""
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
    _record_arrow_query_result(
        table,
        query,
        arrow,
        operation=operation,
        setup_ms=setup_ms,
        execution_ms=execution_ms,
        to_pylist_ms=0.0,
    )
    return arrow

# =============================================================================
# Index construction and term/posting storage
# =============================================================================

def iter_indexed_terms(content: str):
    """Yield ``(term_type, normalized_term, start, end, wrapper_flag)``."""
    for token, start, end in iter_tokens(content):
        yield (
            "normal",
            token,
            int(start),
            int(end),
            wrapper_flag_for_span(content, int(start), int(end)),
        )

    for _normalized_chunk, start, end in iter_additional_tokens(content):
        raw_token = content[int(start):int(end)]
        raw_core = raw_token.rstrip(_ADDITIONAL_TRAILING_PUNCTUATION)
        if not raw_core:
            continue
        canonical = canonicalize_additional_token(raw_core)
        if not canonical or len(canonical) > FTS_MAX_CHUNK_LENGTH:
            continue
        wrapped = _is_wrapped_additional_token(raw_core)
        # Wrapped ordinary words are indexed only under their normal term;
        # the wrapper is recorded per occurrence instead.
        if not is_indexable_additional_token(canonical):
            continue
        span_start = int(start) + (1 if wrapped else 0)
        span_end = span_start + len(canonical)
        yield "additional", canonical, span_start, int(span_end), (
            raw_core[0] + raw_core[-1] if wrapped else ""
        )


def indexed_term_keys(content: str) -> set[TermKey]:
    return {
        (str(term_type), str(term))
        for term_type, term, _start, _end, _flag in iter_indexed_terms(content)
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
    flags_by_term: dict[int, list[str]] = {}
    for term_type, term, start, end, flag in iter_indexed_terms(content):
        term_id = term_to_id[(term_type, term)]
        spans_by_term.setdefault(term_id, []).extend(
            (int(start), int(end - start))
        )
        flags_by_term.setdefault(term_id, []).append(flag)

    return [
        {
            "doc_id": int(doc_id),
            "term_id": int(term_id),
            "positions": np.asarray(flat_positions, dtype=np.int32),
            "wrapper_flags": list(flags_by_term[term_id]),
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
    """Clean rebuild with one read/hash and one tokenization per unique SHA-256."""
    source_paths = collect_paths(paths)
    _notify_progress(progress_callback, "Reading and hashing text files", 0.0)

    if not source_paths:
        (
            docs,
            locations,
            postings,
            terms,
            frequencies,
            fts_docs,
        ) = create_empty_index(db)
        return docs, locations, postings, terms, frequencies, fts_docs, {}, False, {}

    read_workers = min(16, os.cpu_count() or 1, len(source_paths))
    _notify_progress(
        progress_callback,
        f"Reading {len(source_paths)} files with {read_workers} worker thread(s)",
        0.08,
    )

    # Keep only one raw byte buffer per content hash. Duplicate files are read
    # and hashed once, but their bytes are immediately released after the hash
    # has been recorded; they are never tokenized.
    groups: dict[str, dict[str, object]] = {}
    failed_paths: list[str] = []
    for path_str, result in _iter_read_file_workers(
        source_paths,
        max_workers=read_workers,
    ):
        if result is None:
            failed_paths.append(path_str)
            continue
        sha256 = str(result["sha256"])
        group = groups.get(sha256)
        if group is None:
            groups[sha256] = {
                "paths": [str(Path(path_str).absolute())],
                "raw_bytes": result["raw_bytes"],
            }
        else:
            group["paths"].append(str(Path(path_str).absolute()))

    if failed_paths:
        preview = ", ".join(repr(path) for path in failed_paths[:5])
        more = " ..." if len(failed_paths) > 5 else ""
        raise RuntimeError(
            f"Could not read {len(failed_paths)} source file(s): {preview}{more}. "
            "The existing index was not replaced."
        )
    if not groups:
        raise RuntimeError(
            "No source files were successfully read; the existing index was not replaced."
        )

    tokenization_workers = get_tokenization_worker_count(
        task_count=len(groups),
        requested_workers=TOKENIZATION_MAX_WORKERS,
    )
    _notify_progress(
        progress_callback,
        (
            f"Tokenizing {len(groups)} unique content file(s) "
            f"({TOKENIZATION_BACKEND}, {tokenization_workers} physical-core worker(s), "
            f"batch {TOKENIZATION_BATCH_SIZE})"
        ),
        0.35,
    )
    tokenize_records: list[dict] = []
    for sha256, group in sorted(
        groups.items(),
        key=lambda item: str(min(item[1]["paths"], key=str.casefold)).casefold(),
    ):
        group_paths = sorted(group["paths"], key=str.casefold)
        group["paths"] = group_paths
        tokenize_records.append({
            "path": group_paths[0],
            "sha256": sha256,
            "raw_bytes": group.pop("raw_bytes"),
        })

    tokenized_results = _run_tokenize_content_workers(tokenize_records)
    tokenize_records.clear()

    tokenization_failures: list[str] = []
    for record, result in zip(
        sorted(groups.items(), key=lambda item: str(item[1]["paths"][0]).casefold()),
        tokenized_results,
    ):
        sha256, group = record
        if result is None:
            tokenization_failures.append(str(group["paths"][0]))
            continue
        group.update({
            "content": result["content"],
            "word_count": result["word_count"],
            "grouped_terms": result["grouped_terms"],
        })
    del tokenized_results

    if tokenization_failures:
        preview = ", ".join(repr(path) for path in tokenization_failures[:5])
        more = " ..." if len(tokenization_failures) > 5 else ""
        raise RuntimeError(
            f"Could not tokenize {len(tokenization_failures)} unique content file(s): "
            f"{preview}{more}. The existing index was not replaced."
        )

    _notify_progress(progress_callback, "Preparing unique-content documents", 0.48)
    documents: list[dict] = []
    location_rows: list[dict] = []
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: str(item[1]["paths"][0]).casefold(),
    )

    for doc_id, (sha256, group) in enumerate(ordered_groups):
        group_paths = group["paths"]
        representative = group_paths[0]
        documents.append(
            document_row(
                doc_id,
                Path(representative),
                str(group["content"]),
                sha256,
                word_count=int(group["word_count"]),
            )
        )
        location_rows.extend(
            {"doc_id": int(doc_id), "path": path_str}
            for path_str in group_paths
        )

    (
        docs,
        locations,
        postings_table,
        terms_table,
        frequencies_table,
        fts_table,
    ) = create_empty_index(db)

    _notify_progress(progress_callback, "Building unique-content terms", 0.58)
    all_terms: set[TermKey] = set()
    for _sha256, group in ordered_groups:
        all_terms.update(group["grouped_terms"].keys())

    term_to_id: dict[TermKey, int] = {
        term_key: term_id
        for term_id, term_key in enumerate(sorted(all_terms))
    }
    term_rows = [
        {
            "term_id": int(term_id),
            "term_type": term_key[0],
            "term": term_key[1],
            "token_length": int(len(term_key[1])),
        }
        for term_key, term_id in sorted(term_to_id.items(), key=lambda item: item[1])
    ]
    add_rows(terms_table, term_rows, TERMS_SCHEMA)

    postings: list[dict] = []
    frequencies: list[dict] = []
    fts_rows: list[dict] = []
    doc_word_counts: dict[int, int] = {}

    for document, (_sha256, group) in zip(documents, ordered_groups):
        doc_id = int(document["id"])
        grouped_terms = group["grouped_terms"]
        doc_postings = [
            {
                "doc_id": doc_id,
                "term_id": int(term_to_id[term_key]),
                "positions": payload["positions"],
                "wrapper_flags": payload["wrapper_flags"],
            }
            for term_key, payload in sorted(
                grouped_terms.items(),
                key=lambda item: term_to_id[item[0]],
            )
        ]
        postings.extend(doc_postings)
        frequencies.extend(frequency_rows_from_postings(doc_postings))
        doc_word_counts[doc_id] = int(group["word_count"])

        for term_key in sorted(grouped_terms, key=lambda key: term_to_id[key]):
            term = term_key[1]
            if len(term) <= FTS_MAX_CHUNK_LENGTH:
                fts_rows.append({
                    "doc_id": doc_id,
                    "term_id": int(term_to_id[term_key]),
                    FTS_KEY_COLUMN: encode_literal_fts_key(term),
                })

    _notify_progress(progress_callback, "Writing unique-content index", 0.78)
    add_rows(docs, documents, DOCS_SCHEMA)
    add_rows(locations, location_rows, DOCUMENT_LOCATIONS_SCHEMA)
    add_rows(postings_table, postings, POSTINGS_SCHEMA)
    add_rows(frequencies_table, frequencies, TERM_FREQUENCIES_SCHEMA)

    # Create scalar indexes only after the bulk append, avoiding repeated
    # index maintenance while the initial postings/frequency tables are built.
    ensure_search_scalar_indexes(postings_table, frequencies_table)

    _notify_progress(progress_callback, "Writing FTS entries", 0.91)
    add_rows(fts_table, fts_rows, FTS_DOCS_SCHEMA)

    _notify_progress(progress_callback, "Creating FTS index", 0.96)
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
    removed_normal_terms = False
    if obsolete:
        delete_rows_in(index.terms, "term_id", sorted(obsolete))
        for term_id in obsolete:
            key = index.id_to_term.pop(int(term_id), None)
            if key is None:
                index.term_lengths.pop(int(term_id), None)
                continue
            index.term_to_id.pop(key, None)
            term_type, term = key
            if term_type == "normal":
                removed_normal_terms = True
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
        if removed_normal_terms:
            # The lazy suffix buckets contain a reversed copy of the normal
            # vocabulary. Keep them coherent after sync/deletion as well as
            # after insertions.
            index.normal_terms_by_reversed_length.clear()
    return obsolete


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

# =============================================================================
# Full-text-search candidate acceleration
# =============================================================================

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

    metadata_started = time.perf_counter()
    row_limit = max(int(index.fts_docs.count_rows()), 1)
    _record_db_metadata_call(
        "Tantivy candidate row-count limit",
        (time.perf_counter() - metadata_started) * 1000.0,
    )

    arrow, used_score_fallback, projection_setup_ms, execution_ms = (
        _fts_to_arrow_without_score(search, row_limit)
    )
    setup_ms += projection_setup_ms

    started_py = time.perf_counter()
    rows = arrow.to_pylist()
    to_pylist_ms = (time.perf_counter() - started_py) * 1000.0
    _record_arrow_query_result(
        index.fts_docs,
        search,
        arrow,
        operation="Tantivy exact candidate lookup",
        setup_ms=setup_ms,
        execution_ms=execution_ms,
        to_pylist_ms=to_pylist_ms,
        rows_returned=len(rows),
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
    report: SearchReport,
) -> set[int] | None:
    """Use exact candidate acceleration after Python term-type resolution."""
    if not index.fts_enabled:
        return None

    candidate_docs: set[int] | None = None
    all_docs_cache: set[int] | None = None

    def widen_database_candidates(entry: dict, status: str) -> set[int]:
        nonlocal all_docs_cache
        if all_docs_cache is None:
            all_docs_cache = all_document_ids(index)
        entry["status"] = status
        entry["path"] = "database candidate pool for authoritative postings"
        return set(all_docs_cache)

    for query_term, route, variants in zip(
        query_terms, routes, variants_by_term
    ):
        started = time.perf_counter()
        term_type = (
            "additional" if route == "additional"
            else "unindexed" if route == "unindexed"
            else "wildcard" if route == "wildcard"
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

        # A standalone * / ? phrase component is checked positionally between
        # indexed anchors. It must neither expand vocabulary terms nor widen
        # the candidate pool to every document.
        if route == "wildcard":
            entry["status"] = "positional"
            entry["path"] = "anchor-driven source-position wildcard"
            entry["time_ms"] = (time.perf_counter() - started) * 1000.0
            report.tantivy_by_term.append(entry)
            report.tantivy_ms += entry["time_ms"]
            continue

        try:
            if not variants:
                docs_for_keyword = set()
                entry["status"] = "unresolved"
                entry["path"] = "no indexed variant; no literal scan"
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
                    docs_for_keyword = widen_database_candidates(
                        entry,
                        "Tantivy returned no candidates; widened database candidate pool",
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
                    docs_for_keyword = widen_database_candidates(
                        entry,
                        "Resolved-term filter returned no candidates; widened database candidate pool",
                    )
        except Exception as exc:
            docs_for_keyword = widen_database_candidates(
                entry,
                f"Tantivy unavailable; widened database candidate pool ({type(exc).__name__})",
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

    # If the complete query contains only positional wildcard components,
    # there is no indexed anchor and a source scan of all document IDs is
    # unavoidable. Mixed queries retain the candidate intersection above.
    if candidate_docs is None:
        candidate_docs = all_document_ids(index)
    report.candidate_docs_before_positions = len(candidate_docs)
    return candidate_docs

# =============================================================================
# Vocabulary resolution and search planning
# =============================================================================

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


def _terms_with_prefix(bucket: list[str], prefix: str):
    """Yield one sorted length bucket's contiguous prefix range using bisect."""
    position = bisect_left(bucket, prefix)
    while position < len(bucket):
        term = bucket[position]
        if not term.startswith(prefix):
            break
        yield term
        position += 1


def _reversed_normal_terms_by_length(index: "SearchIndex") -> dict[int, list[str]]:
    """Build the suffix lookup lazily, avoiding database-open/load-time cost."""
    if not index.normal_terms_by_reversed_length:
        index.normal_terms_by_reversed_length = {
            token_length: sorted(term[::-1] for term in terms)
            for token_length, terms in index.normal_terms_by_length.items()
        }
    return index.normal_terms_by_reversed_length


def resolve_wildcard_variants(
    index: "SearchIndex",
    query: str,
) -> dict[int, int]:
    """Resolve embedded * / ? patterns without scanning the whole vocabulary.

    Candidate term lengths are bounded first. A literal prefix or suffix then
    selects a contiguous range from the corresponding sorted vocabulary
    bucket; the compiled wildcard regex is applied only to that reduced set.
    The function preserves the existing regex-mode semantics and returns the
    same ``term_id -> edit_distance`` mapping as ``resolve_variants``.
    """
    normalized = query.casefold()
    first_wildcard = min(
        (position for position, char in enumerate(normalized) if char in "*?"),
        default=len(normalized),
    )
    last_wildcard = max(
        (position for position, char in enumerate(normalized) if char in "*?"),
        default=-1,
    )
    prefix = normalized[:first_wildcard]
    suffix = normalized[last_wildcard + 1:]
    minimum_length = len(normalized)  # * consumes at least one, ? exactly one.
    maximum_length = max(index.normal_terms_by_length, default=0)
    pattern = compile_wildcard(normalized)
    if pattern is None or minimum_length > maximum_length:
        return {}

    report = _ACTIVE_DB_QUERY_REPORT.get()
    started = time.perf_counter()
    use_reversed = bool(suffix) and (
        not prefix or len(suffix) > len(prefix)
    )
    reversed_buckets = (
        _reversed_normal_terms_by_length(index) if use_reversed else {}
    )

    examined = 0
    result: dict[int, int] = {}

    for token_length in range(minimum_length, maximum_length + 1):
        original_bucket = index.normal_terms_by_length.get(token_length, ())
        if not original_bucket:
            continue

        if use_reversed:
            reversed_bucket = reversed_buckets.get(token_length, ())
            reverse_prefix = suffix[::-1]
            candidates = (
                reversed_term[::-1]
                for reversed_term in _terms_with_prefix(
                    reversed_bucket,
                    reverse_prefix,
                )
            )
        elif prefix:
            candidates = _terms_with_prefix(original_bucket, prefix)
        else:
            candidates = iter(original_bucket)

        for term in candidates:
            examined += 1
            if prefix and not term.startswith(prefix):
                continue
            if suffix and not term.endswith(suffix):
                continue
            if pattern.fullmatch(term) is None:
                continue
            term_id = index.term_to_id.get(("normal", term))
            if term_id is not None:
                result[int(term_id)] = 0

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if report is not None:
        report.wildcard_candidate_terms_examined += examined
        report.wildcard_candidate_terms_matched += len(result)
        report.wildcard_candidate_lookup_ms += elapsed_ms
    return result


def resolve_variants(
    index: "SearchIndex",
    query_term: str,
    term_type: TermType,
    mode: TermSearchMode,
    config: SearchConfig,
) -> dict[int, int]:
    """Resolve variants strictly within one term type.

    ``both`` merges regex/wildcard matches and fuzzy variants, preferring the
    lowest edit distance when a term is found by both strategies.
    """
    query = query_term.strip()
    if mode == "both":
        merged = resolve_variants(index, query_term, term_type, "regex", config)
        fuzzy_variants = resolve_variants(index, query_term, term_type, "fuzzy", config)
        for term_id, distance in fuzzy_variants.items():
            merged[term_id] = min(merged.get(term_id, distance), distance)
        return merged
    if not query:
        return {}

    normalized = query.casefold()
    if term_type == "normal":
        # Wrapped plain-word queries such as "(hello)" resolve through the
        # normal namespace; canonicalization strips the wrapper pair.
        normalized = _strip_additional_wrapper(normalized)
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
            not _has_regex_syntax_except_wildcards(normalized)
            and "*" not in normalized
            and "?" not in normalized
        ):
            term_id = index.term_to_id.get(("normal", normalized))
            return {int(term_id): 0} if term_id is not None else {}

        # Embedded character wildcards use length + fixed-prefix/suffix
        # candidate selection. Arbitrary regex syntax retains the full
        # vocabulary scan because no safe lexical bounds can be inferred.
        if (
            ("*" in normalized or "?" in normalized)
            and not _has_regex_syntax_except_wildcards(normalized)
        ):
            return resolve_wildcard_variants(index, normalized)

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
    mode: TermSearchMode,
) -> str:
    if route == "unindexed":
        return "unindexed query (no database match)"
    if term_type == "additional":
        if mode == "both":
            return "additional regex+fuzzy term lookup"
        return (
            "additional exact term lookup"
            if mode == "regex"
            else "additional fuzzy term lookup"
        )
    if mode == "fuzzy":
        return "normal fuzzy term lookup"
    if mode == "both":
        return "normal regex+fuzzy term lookup"
    if route == "normal":
        return "normal exact/wildcard/regex term lookup"
    return "term lookup"


def resolve_query_terms(
    index: "SearchIndex",
    query_terms: list[str],
    mode: TermSearchMode,
    config: SearchConfig,
    report: SearchReport,
    term_modes: Sequence[TermSearchMode] | None = None,
    positional_wildcards: Sequence[bool] | None = None,
) -> tuple[list[dict[int, int]], list[QueryRoute]]:
    variants_by_term: list[dict[int, int]] = []
    routes: list[QueryRoute] = []
    effective_modes = normalize_term_modes(
        term_modes if term_modes is not None else (config.term_modes or None),
        len(query_terms),
        mode,
    )
    wildcard_flags = (
        [bool(value) for value in positional_wildcards]
        if positional_wildcards is not None
        else [False] * len(query_terms)
    )
    if len(wildcard_flags) < len(query_terms):
        wildcard_flags.extend([False] * (len(query_terms) - len(wildcard_flags)))
    wildcard_flags = wildcard_flags[:len(query_terms)]

    started_total = time.perf_counter()
    for query_term, term_mode, is_positional_wildcard in zip(
        query_terms, effective_modes, wildcard_flags
    ):
        started = time.perf_counter()
        if is_positional_wildcard:
            variants_by_term.append({})
            routes.append("wildcard")
            report.variant_resolution.append(
                {
                    "query_term": query_term,
                    "term_type": "wildcard",
                    "route": "wildcard",
                    "variants": 0,
                    "path": "positional component",
                    "mode": term_mode,
                    "mode_detail": "anchor-driven source-position wildcard; no term expansion",
                    "time_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
            continue

        route = classify_query(query_term)
        routes.append(route)

        if route == "normal":
            term_type: TermType = "normal"
        elif route == "additional":
            term_type = "additional"
        else:
            term_type = "normal"

        if route == "unindexed":
            variants = {}
            detail = "not represented in the index; no source-text fallback"
        else:
            variants = resolve_variants(
                index,
                query_term,
                term_type,
                term_mode,
                config,
            )
            detail = (
                _variant_mode_detail(route, term_type, term_mode)
                if variants
                else "no indexed variant; no source-text fallback"
            )

        # Database mode is index-only. An unresolved component is an ordinary
        # no-match condition, never a request to scan document text.
        variants_by_term.append(variants)
        report.variant_resolution.append(
            {
                "query_term": query_term,
                "term_type": term_type if route != "unindexed" else "unindexed",
                "route": route,
                "variants": len(variants),
                "path": "terms/postings" if variants else "no indexed variant",
                "mode": term_mode,
                "mode_detail": detail,
                "time_ms": (time.perf_counter() - started) * 1000.0,
            }
        )

    report.variant_resolution_ms = (
        time.perf_counter() - started_total
    ) * 1000.0
    return variants_by_term, routes


def fetch_term_frequencies(
    index: "SearchIndex",
    query_terms: list[str],
    variants_by_term: list[dict[int, int]],
    candidate_docs: set[int],
    report: SearchReport | None = None,
) -> tuple[dict[int, dict[str, int]], int, set[tuple[int, int]]]:
    """Fetch only the needed frequency rows for the current candidate documents."""

    if not candidate_docs:
        return {}, 0, set()

    all_variants = sorted(
        {int(term_id) for variants in variants_by_term for term_id in variants}
    )
    if not all_variants:
        return {}, 0, set()

    frequency_by_doc_term: dict[int, dict[str, int]] = {
        int(doc_id): {} for doc_id in candidate_docs
    }
    normalized_query_terms = [term.casefold() for term in query_terms]
    variant_to_query_terms: dict[int, set[str]] = {}
    for query_term, variants in zip(normalized_query_terms, variants_by_term):
        for term_id in variants:
            variant_to_query_terms.setdefault(int(term_id), set()).add(query_term)

    term_chunk_size = 500
    rows_seen = 0
    frequency_pairs: set[tuple[int, int]] = set()
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
            frequency_pairs.add((doc_id, term_id))
            if report is not None:
                report.runtime_frequency_hits += 1
            rows_seen += 1
            for query_term in variant_to_query_terms.get(term_id, []):
                target = frequency_by_doc_term.setdefault(doc_id, {})
                target[query_term] = target.get(query_term, 0) + count

    if report is not None:
        report.runtime_frequency_lookups += len(candidate_doc_ids) * len(all_variants)
    return frequency_by_doc_term, rows_seen, frequency_pairs


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

# =============================================================================
# Positional postings and proximity matching
# =============================================================================

def fetch_postings(
    index: "SearchIndex",
    term_ids: set[int],
    candidate_docs: set[int],
    report: SearchReport | None = None,
    *,
    required_pairs: set[tuple[int, int]] | None = None,
) -> dict[int, dict[int, PostingOccurrences]]:
    """Fetch decoded positional rows, reusing the bounded in-memory LRU.

    ``required_pairs`` comes from the frequency query and contains only
    ``(doc_id, term_id)`` combinations known to exist. It avoids constructing
    or probing the full Cartesian product when fuzzy variants are involved.
    """
    if not term_ids or not candidate_docs:
        return {}

    stored_term_ids = {int(value) for value in term_ids}
    stored_doc_ids = {int(value) for value in candidate_docs}
    if required_pairs is None:
        expected_pairs: set[tuple[int, int]] | None = None
        pair_keys: list[tuple[int, int]] = []
    else:
        expected_pairs = {
            (int(doc_id), int(term_id))
            for doc_id, term_id in required_pairs
            if int(doc_id) in stored_doc_ids and int(term_id) in stored_term_ids
        }
        pair_keys = sorted(expected_pairs)
        if not pair_keys:
            if report is not None:
                entries, cache_bytes = index.posting_cache.stats()
                report.posting_cache_entries = entries
                report.posting_cache_bytes = cache_bytes
            return {}

    grouped: dict[int, dict[int, PostingOccurrences]] = {}
    cached: dict[tuple[int, int], PostingOccurrences] = {}
    missing_pairs: list[tuple[int, int]] = []

    if expected_pairs is not None:
        cached = index.posting_cache.get_many(pair_keys)
        missing_pairs = [key for key in pair_keys if key not in cached]
        for (doc_id, term_id), occurrence_data in cached.items():
            grouped.setdefault(term_id, {})[doc_id] = occurrence_data
        if report is not None:
            report.posting_cache_hits += len(cached)
            report.posting_cache_misses += len(missing_pairs)
            report.runtime_posting_lookups += len(pair_keys)
            report.runtime_posting_hits += len(cached)

    missing_pair_set = set(missing_pairs)

    if expected_pairs is None:
        query_doc_ids = sorted(stored_doc_ids)
        query_term_ids = sorted(stored_term_ids)
    else:
        query_doc_ids = sorted({doc_id for doc_id, _term_id in missing_pairs})
        query_term_ids = sorted({term_id for _doc_id, term_id in missing_pairs})

    if query_doc_ids and query_term_ids and (
        expected_pairs is None or missing_pairs
    ):
        # Use exact per-term document groups for sparse fuzzy-variant pairs.
        # A compact rectangular predicate remains preferable for dense sets.
        clauses: list[str] = []
        use_exact_pairs = expected_pairs is not None
        if use_exact_pairs:
            docs_by_term: dict[int, list[int]] = {}
            for doc_id, term_id in missing_pairs:
                docs_by_term.setdefault(term_id, []).append(doc_id)
            for term_id in sorted(docs_by_term):
                doc_ids = sorted(set(docs_by_term[term_id]))
                for offset in range(0, len(doc_ids), 1000):
                    doc_chunk = doc_ids[offset : offset + 1000]
                    clauses.append(
                        "(term_id = "
                        + str(int(term_id))
                        + " AND "
                        + where_in("doc_id", doc_chunk)
                        + ")"
                    )
            predicate_text = " OR ".join(clauses)
            # Keep Lance's expression parser inputs bounded. If a fuzzy query
            # expands to many variants, use the simpler indexed IN predicate.
            if len(clauses) > 128 or len(predicate_text) > 180_000:
                use_exact_pairs = False

        if not use_exact_pairs:
            clauses = []
            doc_chunks = [
                query_doc_ids[offset : offset + 1000]
                for offset in range(0, len(query_doc_ids), 1000)
            ]
            term_chunks = [
                query_term_ids[offset : offset + 500]
                for offset in range(0, len(query_term_ids), 500)
            ]
            for doc_chunk in doc_chunks:
                for term_chunk in term_chunks:
                    clauses.append(
                        "("
                        + where_in("doc_id", doc_chunk)
                        + " AND "
                        + where_in("term_id", term_chunk)
                        + ")"
                    )
            predicate_text = " OR ".join(clauses)

        # Normally one query is sufficient. The expression is split only for
        # unusually large fuzzy-variant requests.
        predicate_batches: list[str] = []
        current_clauses: list[str] = []
        current_length = 0
        for clause in clauses:
            addition = len(clause) + (4 if current_clauses else 0)
            if current_clauses and (
                len(current_clauses) >= 128
                or current_length + addition > 180_000
            ):
                predicate_batches.append(" OR ".join(current_clauses))
                current_clauses = []
                current_length = 0
                addition = len(clause)
            current_clauses.append(clause)
            current_length += addition
        if current_clauses:
            predicate_batches.append(" OR ".join(current_clauses))

        fetched_values: dict[tuple[int, int], PostingOccurrences] = {}
        available_columns = set(getattr(index.postings.schema, "names", ()))
        selected_columns = ["doc_id", "term_id", "positions"]
        if "wrapper_flags" in available_columns:
            selected_columns.append("wrapper_flags")

        for predicate in predicate_batches:
            arrow_table = query_arrow_columns(
                index.postings,
                predicate=predicate,
                columns=selected_columns,
                operation="Positional postings query",
            )
            if arrow_table.num_rows == 0:
                continue

            doc_ids_arr = arrow_table.column("doc_id").to_numpy(
                zero_copy_only=False
            )
            term_ids_arr = arrow_table.column("term_id").to_numpy(
                zero_copy_only=False
            )
            pos_chunked = arrow_table.column("positions")
            if isinstance(pos_chunked, pa.ChunkedArray):
                pos_list_array = pa.concat_arrays(pos_chunked.chunks)
            else:
                pos_list_array = pos_chunked

            flat_values = pos_list_array.values.to_numpy(
                zero_copy_only=False
            ).astype(np.int32, copy=False)
            offsets = pos_list_array.offsets.to_numpy(
                zero_copy_only=False
            ).astype(np.int32, copy=False)

            if "wrapper_flags" in arrow_table.column_names:
                flags_chunked = arrow_table.column("wrapper_flags")
                if isinstance(flags_chunked, pa.ChunkedArray):
                    flags_list_array = pa.concat_arrays(flags_chunked.chunks)
                else:
                    flags_list_array = flags_chunked
                nested_started = time.perf_counter()
                flat_flags = flags_list_array.values.to_pylist()
                nested_elapsed_ms = (time.perf_counter() - nested_started) * 1000.0
                active_report = _ACTIVE_DB_QUERY_REPORT.get()
                if active_report is not None:
                    active_report.db_nested_to_pylist_ms += nested_elapsed_ms
                    active_report.db_python_object_creation_ms += nested_elapsed_ms
                flag_offsets = flags_list_array.offsets.to_numpy(
                    zero_copy_only=False
                ).astype(np.int32, copy=False)
            else:
                flat_flags = None
                flag_offsets = None

            for row_index in range(arrow_table.num_rows):
                term_id = int(term_ids_arr[row_index])
                doc_id = int(doc_ids_arr[row_index])
                key = (doc_id, term_id)
                if expected_pairs is not None and key not in expected_pairs:
                    continue
                if expected_pairs is not None and key not in missing_pair_set:
                    continue

                start_idx = int(offsets[row_index])
                end_idx = (
                    int(offsets[row_index + 1])
                    if row_index + 1 < len(offsets)
                    else len(flat_values)
                )
                positions_arr = flat_values[start_idx:end_idx]
                occurrence_count = int(positions_arr.size // 2)
                if flat_flags is not None and flag_offsets is not None:
                    flag_start = int(flag_offsets[row_index])
                    flag_end = int(flag_offsets[row_index + 1])
                    flags_arr = np.asarray(
                        flat_flags[flag_start:flag_end],
                        dtype=str,
                    )
                else:
                    flags_arr = np.full(occurrence_count, "", dtype=str)

                if flags_arr.size != occurrence_count:
                    aligned = np.full(occurrence_count, "", dtype=str)
                    aligned[: min(occurrence_count, flags_arr.size)] = flags_arr[
                        : min(occurrence_count, flags_arr.size)
                    ]
                    flags_arr = aligned

                fetched_values[key] = PostingOccurrences(
                    positions=positions_arr,
                    wrapper_flags=flags_arr,
                )

        if fetched_values:
            cached_values = index.posting_cache.put_many(fetched_values)
            for (doc_id, term_id), value in fetched_values.items():
                occurrence_data = cached_values.get((doc_id, term_id), value)
                grouped.setdefault(term_id, {})[doc_id] = occurrence_data
            if report is not None:
                report.runtime_posting_hits += len(fetched_values)

    if report is not None:
        if expected_pairs is None:
            report.runtime_posting_lookups += len(stored_doc_ids) * len(stored_term_ids)
            report.runtime_posting_hits += sum(len(doc_map) for doc_map in grouped.values())
        entries, cache_bytes = index.posting_cache.stats()
        report.posting_cache_entries = entries
        report.posting_cache_bytes = cache_bytes

    for term_id in stored_term_ids:
        grouped.setdefault(term_id, {})
    return grouped

def build_match_arrays(
    variants: dict[int, int],
    postings: dict[int, dict[int, PostingOccurrences]],
    required_wrapper: str | None = None,
    candidate_doc_ids: set[int] | None = None,
) -> dict[int, MatchArrays]:
    """Merge indexed variants for selected documents and optional wrappers."""

    chunks: dict[
        int,
        list[tuple[np.ndarray, np.ndarray, np.ndarray, int, int]],
    ] = {}

    for term_id, edit_distance in variants.items():
        for doc_id, occurrence_data in postings.get(term_id, {}).items():
            if candidate_doc_ids is not None and doc_id not in candidate_doc_ids:
                continue
            # Accepting the former positions-only shape here keeps this helper
            # compatible with small external integrations and unit fixtures.
            if isinstance(occurrence_data, PostingOccurrences):
                positions = occurrence_data.positions
                wrapper_flags = occurrence_data.wrapper_flags
            else:  # pragma: no cover - compatibility path
                positions = np.asarray(occurrence_data, dtype=np.int32)
                wrapper_flags = np.full(positions.size // 2, "", dtype=str)
            if positions.size == 0:
                continue
            starts = positions[0::2]
            lengths = positions[1::2]
            ends = starts + lengths
            if wrapper_flags.size != starts.size:
                aligned = np.full(starts.size, "", dtype=str)
                aligned[: min(starts.size, wrapper_flags.size)] = wrapper_flags[
                    : min(starts.size, wrapper_flags.size)
                ]
                wrapper_flags = aligned
            if required_wrapper is not None:
                occurrence_mask = wrapper_flags == required_wrapper
                if not np.any(occurrence_mask):
                    continue
                starts = starts[occurrence_mask]
                ends = ends[occurrence_mask]
                wrapper_flags = wrapper_flags[occurrence_mask]
            chunks.setdefault(doc_id, []).append(
                (
                    np.asarray(starts, dtype=np.int32),
                    np.asarray(ends, dtype=np.int32),
                    np.asarray(wrapper_flags, dtype=str),
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
        wrapper_flags = np.concatenate([piece[2] for piece in pieces]).astype(str, copy=False)
        term_ids = np.concatenate(
            [np.full(piece[0].size, piece[3], dtype=np.int32) for piece in pieces]
        )
        edit_distances = np.concatenate(
            [
                np.full(piece[0].size, piece[4], dtype=np.int32)
                for piece in pieces
            ]
        )

        # One physical occurrence belongs to one indexed term. In fuzzy mode,
        # multiple accepted query variants can converge on the same coordinates.
        # Keep the closest accepted variant at each coordinate.
        order = np.lexsort((term_ids, edit_distances, ends, starts))
        starts = starts[order]
        ends = ends[order]
        wrapper_flags = wrapper_flags[order]
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
            wrapper_flags=wrapper_flags[keep],
        )

    return result


def build_matches_for_document(
    query_term: str,
    matches: MatchArrays,
    id_to_term: dict[int, tuple[str, str]],
    mode: TermSearchMode,
) -> list[dict]:
    if len(matches) == 0:
        return []

    query_len = len(canonical_query_term(query_term).casefold())
    output: list[dict] = []
    for occurrence_index, (start, end, term_id, edit_distance, wrapper_flag) in enumerate(zip(
        matches.starts.tolist(),
        matches.ends.tolist(),
        matches.term_ids.tolist(),
        matches.edit_distances.tolist(),
        matches.wrapper_flags.tolist(),
    )):
        if int(term_id) < 0:
            indexed_term = canonical_query_term(query_term)
        else:
            indexed_term = id_to_term[int(term_id)][1]
        component_annotations = (
            matches.component_annotations[occurrence_index]
            if matches.component_annotations
            else []
        )
        if component_annotations:
            component_scores: list[float] = []
            indexed_components: list[str] = []
            for annotation in component_annotations:
                component_query = str(annotation.get("query_term", ""))
                component_term_id = int(annotation.get("term_id", FALLBACK_TERM_ID))
                component_edit = int(annotation.get("edit_distance", 0))
                component_indexed = (
                    canonical_query_term(component_query)
                    if component_term_id < 0
                    else id_to_term[component_term_id][1]
                )
                component_length = max(
                    len(canonical_query_term(component_query).casefold()),
                    len(component_indexed),
                )
                component_scores.append(
                    100.0
                    if component_term_id < 0 or mode == "regex" or component_length == 0
                    else 100.0 * (1.0 - component_edit / component_length)
                )
                indexed_components.append(component_indexed)
            score = sum(component_scores) / len(component_scores)
            indexed_term = " ".join(indexed_components)
        else:
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
                "wrapper_flag": str(wrapper_flag),
                "component_annotations": component_annotations,
            }
        )
    return output


def distance_between(first: dict, second: dict) -> int:
    if first["end"] <= second["start"]:
        return second["start"] - first["end"]
    if second["end"] <= first["start"]:
        return first["start"] - second["end"]
    return 0


_LITERAL_ENGLISH_GERMAN_LOWER_TRANSLATION = str.maketrans(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜẞ", "abcdefghijklmnopqrstuvwxyzäöüß"
)


def find_proximity_chains_numpy(
    matches_by_term: list[MatchArrays],
    query_terms: list[str],
    id_to_term: dict[int, tuple[str, str]],
    config: SearchConfig,
    anchor_term: object | None = None,
    case_sensitive: Sequence[bool] | None = None,
    literal_english_german_casefold: bool = False,
) -> list[list[dict]]:
    """Find proximity chains using flat NumPy arrays.

    Every query-term occurrence is represented by numeric arrays.  Connected
    components are detected from vectorized gap arithmetic; the final chain is
    materialized back to dictionaries only for the public ``SearchHit`` output.
    """

    if not matches_by_term or any(len(matches) == 0 for matches in matches_by_term):
        return []

    # Required multiplicities are based on normalized user keywords. Duplicate
    # GUI slots therefore require separate physical occurrences. Literal mode
    # supplies English/German equivalence so proximity grouping follows the
    # same UTF-8 byte-folding rules as literal matching; database mode keeps
    # its existing Unicode casefold behavior.
    case_flags = normalize_case_flags(case_sensitive, len(query_terms))

    def query_identity(query_term: str, sensitive: bool) -> str:
        if literal_english_german_casefold:
            if sensitive:
                return query_term
            return query_term.translate(_LITERAL_ENGLISH_GERMAN_LOWER_TRANSLATION)
        return query_term.casefold()

    required_counts: dict[tuple[str, bool], int] = {}
    for query_term, sensitive in zip(query_terms, case_flags):
        key = (query_identity(query_term, bool(sensitive)), bool(sensitive))
        required_counts[key] = required_counts.get(key, 0) + 1

    unique_terms = list(required_counts)
    term_code = {term: code for code, term in enumerate(unique_terms)}
    required_per_code = np.asarray(
        [required_counts[term] for term in unique_terms],
        dtype=np.int32,
    )
    display_terms: dict[tuple[str, bool], str] = {}
    for query_term, sensitive in zip(query_terms, case_flags):
        display_terms.setdefault(
            (query_identity(query_term, bool(sensitive)), bool(sensitive)),
            query_term,
        )

    starts_parts: list[np.ndarray] = []
    ends_parts: list[np.ndarray] = []
    term_code_parts: list[np.ndarray] = []
    indexed_term_parts: list[np.ndarray] = []
    edit_parts: list[np.ndarray] = []
    wrapper_parts: list[np.ndarray] = []
    component_annotation_parts: list[list[list[dict]]] = []

    for query_term, matches, sensitive in zip(query_terms, matches_by_term, case_flags):
        code = term_code[(query_identity(query_term, bool(sensitive)), bool(sensitive))]
        if len(matches) == 0:
            continue
        starts_parts.append(matches.starts)
        ends_parts.append(matches.ends)
        term_code_parts.append(
            np.full(len(matches), code, dtype=np.int32)
        )
        indexed_term_parts.append(matches.term_ids)
        edit_parts.append(matches.edit_distances)
        wrapper_parts.append(matches.wrapper_flags)
        if matches.component_annotations:
            component_annotation_parts.extend(matches.component_annotations)
        else:
            component_annotation_parts.extend([[] for _ in range(len(matches))])

    if not starts_parts:
        return []

    starts = np.concatenate(starts_parts).astype(np.int32, copy=False)
    ends = np.concatenate(ends_parts).astype(np.int32, copy=False)
    codes = np.concatenate(term_code_parts).astype(np.int16, copy=False)
    indexed_term_ids = np.concatenate(indexed_term_parts).astype(np.int32, copy=False)
    edit_distances = np.concatenate(edit_parts).astype(np.int32, copy=False)
    wrapper_flags = np.concatenate(wrapper_parts).astype(str, copy=False)
    component_annotations = component_annotation_parts

    # Sort by position, then normalized query code. Duplicate query slots for
    # the same normalized keyword are intentionally adjacent and deduplicated.
    order = np.lexsort((edit_distances, indexed_term_ids, codes, ends, starts))
    starts = starts[order]
    ends = ends[order]
    codes = codes[order]
    indexed_term_ids = indexed_term_ids[order]
    edit_distances = edit_distances[order]
    wrapper_flags = wrapper_flags[order]
    component_annotations = [component_annotations[int(index)] for index in order.tolist()]

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
    wrapper_flags = wrapper_flags[keep]
    component_annotations = [
        annotation
        for annotation, keep_value in zip(component_annotations, keep.tolist())
        if keep_value
    ]

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

    anchor_norm = query_identity(str(anchor_term), False) if anchor_term is not None else None
    planned_anchor_codes = (
        [
            term_code[(query_identity(query_term, bool(sensitive)), bool(sensitive))]
            for query_term, sensitive in zip(query_terms, case_flags)
            if query_identity(query_term, bool(sensitive)) == anchor_norm
        ]
        if anchor_norm is not None
        else []
    )
    event_counts = np.bincount(
        codes.astype(np.int32),
        minlength=len(unique_terms),
    )
    if planned_anchor_codes:
        # The search planner chooses an anchor from term-frequency rows. Map
        # that normalized term back to its actual (term, case-mode) event code.
        # If the same text was entered with different case modes, use the
        # instance with the fewest surviving positional occurrences.
        anchor_code = min(
            planned_anchor_codes,
            key=lambda code: (int(event_counts[code]), code),
        )
    else:
        # Phrase-internal anchors and fallback callers may not map to a
        # top-level query slot. Select the rarest surviving event stream.
        anchor_code = min(
            range(len(unique_terms)),
            key=lambda code: (int(event_counts[code]), unique_terms[code]),
        )

    chains: list[list[dict]] = []

    for component_start, component_end in zip(
        component_starts.tolist(), component_ends.tolist()
    ):
        component_codes = codes[component_start : component_end + 1]
        # Each code's local event indexes are naturally sorted because the
        # component itself is position-sorted. The rarest term is the pivot:
        # its occurrence list bounds how many complete groups can exist.
        positions_by_code = [
            np.flatnonzero(component_codes == code).astype(np.int32, copy=False)
            for code in range(len(unique_terms))
        ]
        anchor_positions = positions_by_code[anchor_code]
        if anchor_positions.size < int(required_per_code[anchor_code]):
            continue

        # Preserve the previous greedy grouping exactly. Previously this was
        # a Python loop over every event, incrementing a count for each term
        # and clearing all counts whenever a complete keyword group appeared.
        # The first complete endpoint is instead the furthest of each term's
        # required-th occurrence after block_start. Searchsorted finds those
        # endpoints directly, so work is tied to anchor occurrences/groups
        # rather than every high-frequency occurrence in the component.
        block_start = 0
        last_complete_index = -1
        while True:
            anchor_offset = int(
                np.searchsorted(anchor_positions, block_start, side="left")
            )
            anchor_required_index = (
                anchor_offset + int(required_per_code[anchor_code]) - 1
            )
            if anchor_required_index >= anchor_positions.size:
                break

            completion_index = int(anchor_positions[anchor_required_index])
            group_complete = True
            for code_index, code_positions in enumerate(positions_by_code):
                if code_index == anchor_code:
                    continue
                required = int(required_per_code[code_index])
                occurrence_offset = int(
                    np.searchsorted(code_positions, block_start, side="left")
                )
                required_index = occurrence_offset + required - 1
                if required_index >= code_positions.size:
                    group_complete = False
                    break
                completion_index = max(
                    completion_index,
                    int(code_positions[required_index]),
                )

            if not group_complete:
                break
            last_complete_index = completion_index
            block_start = completion_index + 1

        if last_complete_index < 0:
            continue

        absolute_end = component_start + last_complete_index
        chain_starts = starts[component_start : absolute_end + 1]
        chain_ends = ends[component_start : absolute_end + 1]
        chain_codes = codes[component_start : absolute_end + 1]
        chain_term_ids = indexed_term_ids[component_start : absolute_end + 1]
        chain_edits = edit_distances[component_start : absolute_end + 1]
        chain_wrappers = wrapper_flags[component_start : absolute_end + 1]

        chain: list[dict] = []
        for local_index, (start, end, code, term_id, edit_distance, wrapper_flag) in enumerate(zip(
            chain_starts.tolist(),
            chain_ends.tolist(),
            chain_codes.tolist(),
            chain_term_ids.tolist(),
            chain_edits.tolist(),
            chain_wrappers.tolist(),
        )):
            query_term = display_terms[unique_terms[int(code)]]
            if int(term_id) < 0:
                indexed_term = canonical_query_term(query_term)
            else:
                indexed_term = id_to_term[int(term_id)][1]
            occurrence_annotations = component_annotations[
                component_start + local_index
            ]
            if occurrence_annotations:
                component_scores: list[float] = []
                indexed_components: list[str] = []
                for annotation in occurrence_annotations:
                    component_query = str(annotation.get("query_term", ""))
                    component_term_id = int(annotation.get("term_id", FALLBACK_TERM_ID))
                    component_edit = int(annotation.get("edit_distance", 0))
                    component_indexed = (
                        canonical_query_term(component_query)
                        if component_term_id < 0
                        else id_to_term[component_term_id][1]
                    )
                    component_length = max(
                        len(canonical_query_term(component_query).casefold()),
                        len(component_indexed),
                    )
                    component_scores.append(
                        100.0
                        if component_term_id < 0 or component_length == 0
                        else 100.0 * (1.0 - component_edit / component_length)
                    )
                    indexed_components.append(component_indexed)
                score = sum(component_scores) / len(component_scores)
                indexed_term = " ".join(indexed_components)
            else:
                max_len = max(
                    len(canonical_query_term(query_term).casefold()),
                    len(indexed_term),
                )
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
                    "wrapper_flag": str(wrapper_flag),
                    "component_annotations": occurrence_annotations,
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

# =============================================================================
# Snippet extraction and document retrieval
# =============================================================================

MATCH_MARKER_OPEN = "\ue000"

MATCH_MARKER_CLOSE = "\ue001"

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


def fetch_indexed_document_contents(
    index: "SearchIndex",
    doc_ids: Sequence[int],
) -> dict[int, dict]:
    """Fetch indexed text snapshots for phrase verification.

    Positional offsets refer to the indexed snapshot, not to a potentially
    modified source file. Reading DOCS.content also avoids reopening every
    candidate file during phrase searches; final result documents still use the
    normal hash-verified ``fetch_documents`` path.
    """
    if not doc_ids:
        return {}
    rows = query_rows_in(
        index.docs,
        "id",
        sorted(int(doc_id) for doc_id in doc_ids),
        ["id", "content"],
        operation="Phrase source-text lookup",
    )
    return {
        int(row["id"]): {"content": str(row.get("content", "") or "")}
        for row in rows
    }


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


def fetch_literal_search_documents(
    index: "SearchIndex",
    doc_ids: Sequence[int],
    report: SearchReport | None = None,
) -> dict[int, dict]:
    """Read physical source files for literal search without hashing them.

    Literal mode deliberately searches the current physical source text. The
    normal result-fetch path hashes every file to confirm it still matches the
    indexed snapshot; that verification is unnecessary for this explicit
    original-text mode and adds a full pass over every file's bytes. If no
    physical location can be read, use the indexed content snapshot as the
    same conservative fallback used by normal document retrieval.
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

        locations = list(metadata.get("locations", []))
        candidates = locations or [str(metadata.get("path", ""))]
        loaded = False
        last_error: Exception | None = None

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate))
            try:
                raw_bytes = path.read_bytes()
                byte_count = len(raw_bytes)
                document = dict(metadata)
                document["path"] = str(path)
                document["locations"] = locations
                # Keep physical text as bytes. Most documents will be rejected
                # by byte matching and never need UTF-8 decoding at all.
                document["raw_content"] = raw_bytes
                documents[doc_id] = document
                loaded = True

                if report is not None:
                    report.runtime_source_file_reads += 1
                    report.runtime_source_bytes_read += byte_count
                break
            except Exception as exc:
                last_error = exc

        if not loaded:
            logger.warning(
                "Using LanceDB literal-search fallback for doc_id=%s locations=%s: %s",
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
            operation="Literal-search indexed-content fallback",
        )
        for row in rows:
            doc_id = int(row["id"])
            metadata = index.document_metadata.get(doc_id, {})
            row["locations"] = list(metadata.get("locations", []))
            if row["locations"]:
                row["path"] = row["locations"][0]
            documents[doc_id] = row

    return documents

# =============================================================================
# Database maintenance
# =============================================================================

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

# =============================================================================
# Incremental synchronization
# =============================================================================

def sync_index(
    index: "SearchIndex",
    paths: list[Path],
    *,
    remove_paths: set[str] | None = None,
    trusted_unchanged_paths: set[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, int]:
    """Incrementally reconcile source paths and unique SHA-256 documents.

    All untrusted current files are hashed first. Tokenization is performed only
    once for each SHA-256 that is not already represented in the database. Thus
    unchanged files, renamed/moved files, and duplicate copies reuse their existing
    postings and term IDs without rebuilding their content index.
    """
    remove_paths = {str(Path(path).absolute()) for path in (remove_paths or set())}
    trusted_unchanged_paths = {
        str(Path(path).absolute()) for path in (trusted_unchanged_paths or set())
    }
    source_paths = collect_paths(paths)
    scanned_path_map = {str(path.absolute()): path for path in source_paths}
    scanned_path_strings = set(scanned_path_map)

    _notify_progress(progress_callback, "Loading current index state", 0.05)

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
    orphan_locations = [
        path_str for path_str, doc_id in old_location_by_path.items()
        if doc_id not in old_docs_by_id
    ]
    if orphan_locations:
        raise RuntimeError(
            "The index contains location records for missing content documents: "
            + ", ".join(repr(path) for path in orphan_locations[:5])
        )

    # Keep indexed locations that this call does not manage. For a full folder
    # reconciliation, remove_paths contains the managed old paths that vanished.
    desired_location_data: dict[str, dict[str, object]] = {}
    for path_str, old_doc_id in old_location_by_path.items():
        if path_str in remove_paths or path_str in scanned_path_strings:
            continue
        old_doc = old_docs_by_id.get(old_doc_id)
        if old_doc is None:
            raise RuntimeError(
                f"Location {path_str!r} references missing doc_id {old_doc_id}."
            )
        desired_location_data[path_str] = {
            "path": Path(path_str),
            "sha256": str(old_doc["sha256"]),
            "content": None,
        }

    # Reuse hashes for metadata-identical paths. For all other paths, read the
    # bytes and hash them in the same operation. Keep bytes only for the first
    # path whose content hash is not already indexed; duplicates are not read a
    # second time and never reach the tokenizer.
    hash_by_path: dict[str, str] = {}
    paths_to_read: list[Path] = []
    for path_str, path in scanned_path_map.items():
        old_doc_id = old_location_by_path.get(path_str)
        if path_str in trusted_unchanged_paths and old_doc_id is not None:
            old_doc = old_docs_by_id.get(old_doc_id)
            if old_doc is None:
                raise RuntimeError(
                    f"Location {path_str!r} references missing doc_id {old_doc_id}."
                )
            hash_by_path[path_str] = str(old_doc["sha256"])
        else:
            paths_to_read.append(path)

    _notify_progress(
        progress_callback,
        f"Reading and checking SHA-256 for {len(paths_to_read)} file(s)",
        0.12,
    )
    raw_bytes_by_hash: dict[str, bytes] = {}
    new_hash_representatives: dict[str, str] = {}
    for path_str, read_result in _iter_read_file_workers(paths_to_read, max_workers=16):
        if read_result is None:
            raise RuntimeError(
                f"Could not read source file {path_str!r}; the existing index was left unchanged."
            )
        sha256 = str(read_result["sha256"])
        hash_by_path[path_str] = sha256
        if sha256 not in old_docs_by_sha and sha256 not in new_hash_representatives:
            new_hash_representatives[sha256] = path_str
            raw_bytes_by_hash[sha256] = read_result["raw_bytes"]
        # For known hashes and duplicate new files the raw bytes fall out of scope
        # immediately; only one representative buffer per new hash is retained.

    for path_str, path in scanned_path_map.items():
        sha256 = hash_by_path[path_str]
        desired_location_data[path_str] = {
            "path": path,
            "sha256": sha256,
            "content": None,
        }

    tokenize_records = [
        {
            "path": path_str,
            "sha256": sha256,
            "raw_bytes": raw_bytes_by_hash.pop(sha256),
        }
        for sha256, path_str in sorted(
            new_hash_representatives.items(), key=lambda item: item[1].casefold()
        )
    ]
    _notify_progress(
        progress_callback,
        f"Tokenizing {len(tokenize_records)} new/changed content file(s)",
        0.28,
    )
    tokenized_by_hash: dict[str, dict] = {}
    if tokenize_records:
        processed_results = _run_tokenize_content_workers(tokenize_records)
        tokenize_records.clear()

        for result in processed_results:
            if result is None:
                raise RuntimeError(
                    "A new/changed source file could not be tokenized; "
                    "the existing index has been left unchanged."
                )
            actual_sha = str(result["sha256"])
            tokenized_by_hash[actual_sha] = result
        del processed_results

    _notify_progress(progress_callback, "Grouping unique content", 0.42)

    groups: dict[str, dict[str, object]] = {}
    for path_str in sorted(desired_location_data, key=str.casefold):
        record = desired_location_data[path_str]
        sha256 = str(record["sha256"])
        group = groups.get(sha256)
        if group is None:
            tokenized = tokenized_by_hash.get(sha256, {})
            groups[sha256] = {
                "paths": [path_str],
                "content": record.get("content") or tokenized.get("content"),
                "grouped_terms": record.get(
                    "grouped_terms", tokenized.get("grouped_terms", {})
                ),
                "word_count": record.get("word_count", tokenized.get("word_count", 0)),
            }
        else:
            group["paths"].append(path_str)
            if group["content"] is None and record.get("content") is not None:
                group["content"] = record["content"]
                group["grouped_terms"] = record["grouped_terms"]
                group["word_count"] = record["word_count"]
            elif group["content"] is None and sha256 in tokenized_by_hash:
                tokenized = tokenized_by_hash[sha256]
                group["content"] = tokenized["content"]
                group["grouped_terms"] = tokenized["grouped_terms"]
                group["word_count"] = tokenized["word_count"]

    current_hashes = set(groups)

    # Keep IDs for hashes already in the index. For a one-to-one modification
    # where a path's old content has disappeared entirely, reuse that old ID for
    # the new hash. This replaces just that document's postings instead of adding
    # a new ID and then deleting the old one.
    doc_id_by_hash: dict[str, int] = {
        sha256: int(old_doc["id"])
        for sha256, old_doc in old_docs_by_sha.items()
        if sha256 in current_hashes
    }
    assigned_ids = set(doc_id_by_hash.values())
    reuse_candidates_by_hash: dict[str, list[tuple[str, int]]] = {}
    for path_str, record in desired_location_data.items():
        new_sha = str(record["sha256"])
        old_doc_id = old_location_by_path.get(path_str)
        if old_doc_id is None or new_sha in old_docs_by_sha:
            continue
        old_doc = old_docs_by_id.get(old_doc_id)
        if old_doc is None:
            continue
        old_sha = str(old_doc["sha256"])
        if old_sha != new_sha and old_sha not in current_hashes:
            reuse_candidates_by_hash.setdefault(new_sha, []).append((path_str, old_doc_id))

    next_id = max(old_docs_by_id, default=-1) + 1
    for sha256 in sorted(
        (value for value in current_hashes if value not in doc_id_by_hash),
        key=lambda value: str(groups[value]["paths"][0]).casefold(),
    ):
        candidates = sorted(
            reuse_candidates_by_hash.get(sha256, []), key=lambda item: item[0].casefold()
        )
        reusable_id = next(
            (
                int(old_id)
                for _path, old_id in candidates
                if int(old_id) not in assigned_ids
            ),
            None,
        )
        if reusable_id is not None:
            doc_id_by_hash[sha256] = reusable_id
            assigned_ids.add(reusable_id)
        else:
            doc_id_by_hash[sha256] = int(next_id)
            assigned_ids.add(int(next_id))
            next_id += 1

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
            # An existing SHA can reuse the stored DB row. A truly new SHA must
            # have been tokenized above, otherwise something went wrong.
            if sha256 not in old_docs_by_sha:
                raise RuntimeError(f"Cannot index content hash {sha256}: token data is missing.")
            continue

        representative_path = Path(str(group["paths"][0]))
        new_row = document_row(
            doc_id,
            representative_path,
            str(content),
            sha256,
            word_count=int(group["word_count"]),
        )
        if old_doc is None:
            added_docs.append(new_row)
        else:
            changed_docs.append(new_row)
            changed_ids.add(doc_id)

    assigned_ids = set(doc_id_by_hash.values())
    removed_ids = set(old_docs_by_id) - assigned_ids
    affected_ids = set(removed_ids) | changed_ids

    desired_path_to_doc = {
        path_str: int(doc_id_by_hash[str(record["sha256"])])
        for path_str, record in desired_location_data.items()
    }
    locations_removed_paths = {
        path_str
        for path_str, old_id in old_location_by_path.items()
        if path_str not in desired_path_to_doc or desired_path_to_doc[path_str] != old_id
    }
    locations_added = sum(
        1 for path_str, new_id in desired_path_to_doc.items()
        if old_location_by_path.get(path_str) != new_id
    )
    locations_removed = len(locations_removed_paths)

    # A missing old path paired with a new path of the same hash is a move/rename;
    # adding a duplicate while its old path remains is just another location.
    missing_old_by_sha: dict[str, list[str]] = {}
    for old_path, old_doc_id in old_location_by_path.items():
        if old_path in desired_path_to_doc or old_path not in remove_paths:
            continue
        old_doc = old_docs_by_id.get(old_doc_id)
        if old_doc is not None:
            missing_old_by_sha.setdefault(str(old_doc["sha256"]), []).append(old_path)
    new_paths_by_sha: dict[str, list[str]] = {}
    for path_str, record in desired_location_data.items():
        if path_str not in old_location_by_path:
            new_paths_by_sha.setdefault(str(record["sha256"]), []).append(path_str)
    renamed_count = sum(
        min(len(missing_paths), len(new_paths_by_sha.get(sha256, [])))
        for sha256, missing_paths in missing_old_by_sha.items()
    )
    changed_path_count = sum(
        1
        for path_str, record in desired_location_data.items()
        if path_str in old_location_by_path
        and str(old_docs_by_id[old_location_by_path[path_str]]["sha256"])
        != str(record["sha256"])
    )

    _notify_progress(progress_callback, "Updating locations and unique-content index", 0.60)
    if affected_ids or changed_docs or added_docs:
        # Changed/deleted content can reuse doc_id values. Drop decoded rows
        # before the persistent postings are mutated so no stale occurrence
        # arrays can be served by a later search.
        index.posting_cache.clear()
    locations_to_add = [
        {"doc_id": int(doc_id), "path": path_str}
        for path_str, doc_id in sorted(desired_path_to_doc.items(), key=lambda item: item[0].casefold())
        if old_location_by_path.get(path_str) != doc_id
    ]

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

    # Location maintenance is independent of content-index maintenance. In
    # particular, a move/rename has no affected content document, but its old path
    # still must be removed before the new location is inserted.
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        if locations_removed_paths:
            futures.append(
                executor.submit(
                    delete_rows_in, index.locations, "path", sorted(locations_removed_paths)
                )
            )
        if affected_ids:
            futures.append(executor.submit(delete_rows_in, index.postings, "doc_id", sorted(affected_ids)))
            futures.append(executor.submit(delete_rows_in, index.frequencies, "doc_id", sorted(affected_ids)))
            futures.append(executor.submit(delete_rows_in, index.fts_docs, "doc_id", sorted(affected_ids)))
        if removed_ids or changed_ids:
            futures.append(
                executor.submit(delete_rows_in, index.docs, "id", sorted(removed_ids | changed_ids))
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        if locations_to_add:
            futures.append(
                executor.submit(add_rows, index.locations, locations_to_add, DOCUMENT_LOCATIONS_SCHEMA)
            )
        if changed_docs or added_docs:
            futures.append(
                executor.submit(add_rows, index.docs, changed_docs + added_docs, DOCS_SCHEMA)
            )
        for future in concurrent.futures.as_completed(futures):
            future.result()

    pending_terms: set[TermKey] = set()
    for document in changed_docs + added_docs:
        sha256 = str(document["sha256"])
        group = groups[sha256]
        pending_terms.update(group["grouped_terms"].keys())

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
            index.normal_terms_by_length if term_key[0] == "normal"
            else index.additional_terms_by_length
        )
        target.setdefault(len(term_key[1]), []).append(term_key[1])
        target[len(term_key[1])].sort()
        new_terms.append((term_key, term_id))

    if any(term_key[0] == "normal" for term_key, _term_id in new_terms):
        # Incremental synchronization can extend the vocabulary while this
        # SearchIndex object remains alive. Invalidate the lazy suffix map so
        # subsequent wildcard lookups also see newly indexed normal terms.
        index.normal_terms_by_reversed_length.clear()

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

    _notify_progress(progress_callback, "Building unique-content postings", 0.80)
    new_postings: list[dict] = []
    new_frequencies: list[dict] = []
    new_fts_rows: list[dict] = []
    for document in changed_docs + added_docs:
        doc_id = int(document["id"])
        sha256 = str(document["sha256"])
        group = groups[sha256]
        grouped_terms = group["grouped_terms"]
        document_postings = [
            {
                "doc_id": int(doc_id),
                "term_id": int(index.term_to_id[term_key]),
                "positions": payload["positions"],
                "wrapper_flags": payload["wrapper_flags"],
            }
            for term_key, payload in sorted(
                grouped_terms.items(),
                key=lambda item: index.term_to_id[item[0]],
            )
        ]
        new_postings.extend(document_postings)
        new_frequencies.extend(frequency_rows_from_postings(document_postings))
        index.doc_word_counts[doc_id] = int(group["word_count"])

        doc_term_keys = {
            term_key
            for term_key in grouped_terms
            if len(term_key[1]) <= FTS_MAX_CHUNK_LENGTH
        }
        for key in sorted(doc_term_keys, key=lambda item: index.term_to_id[item]):
            new_fts_rows.append({
                "doc_id": int(doc_id),
                "term_id": int(index.term_to_id[key]),
                FTS_KEY_COLUMN: encode_literal_fts_key(key[1]),
            })
        affected_term_ids.update(int(row["term_id"]) for row in document_postings)

    _notify_progress(progress_callback, "Writing unique-content postings", 0.90)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        if new_postings:
            futures.append(executor.submit(add_rows, index.postings, new_postings, POSTINGS_SCHEMA))
        if new_frequencies:
            futures.append(executor.submit(add_rows, index.frequencies, new_frequencies, TERM_FREQUENCIES_SCHEMA))
        if new_fts_rows:
            futures.append(executor.submit(add_rows, index.fts_docs, new_fts_rows, FTS_DOCS_SCHEMA))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    remove_unused_terms(index, affected_term_ids)
    if changed_docs or added_docs or affected_ids:
        _notify_progress(progress_callback, "Refreshing FTS index", 0.95)
        index.fts_enabled = refresh_fts_index(index.fts_docs)

    index.document_metadata, index.doc_word_counts = build_runtime_search_caches(
        index.docs, index.locations
    )
    _notify_progress(progress_callback, "Index synchronization complete", 1.0)

    return {
        "added": len(added_docs),
        "changed": max(len(changed_docs), int(changed_path_count)),
        "removed": len(removed_ids),
        "unchanged": max(0, len(groups) - len(added_docs) - len(changed_docs)),
        "renamed": int(renamed_count),
        "locations_added": int(locations_added),
        "locations_removed": int(locations_removed),
        "unique_documents": int(len(groups)),
        "physical_files": int(len(desired_path_to_doc)),
    }

# =============================================================================
# Literal source-text search
# =============================================================================

# Each worker loads this many complete document texts into memory before
# matching them. At most one batch per worker is resident at a time.
LITERAL_SEARCH_FILES_PER_WORKER_BATCH = 100


def search_literal_text(
    index: "SearchIndex",
    query_terms: list[str],
    case_flags: Sequence[bool],
    config: SearchConfig,
    max_results: int | None,
    report: SearchReport,
    progress_callback: ProgressCallback | None = None,
    selected_doc_ids: set[int] | None = None,
    started_total: float | None = None,
) -> tuple[list[SearchHit], SearchReport]:
    """Search original source text in parallel, loading 100 files per worker.

    This path intentionally bypasses vocabulary lookup, FTS candidates,
    postings, wildcard expansion, regex syntax interpretation, and fuzzy
    matching. Each worker loads its complete 100-file batch as physical source
    bytes and searches those bytes directly. Case-insensitive matching folds
    English ASCII letters plus German Ä/ä, Ö/ö, Ü/ü and ẞ/ß directly in UTF-8
    bytes; ß and
    ss remain distinct. Only documents containing every query term are
    decoded, so byte offsets can be converted to character offsets for
    proximity checks and result snippets. The worker pool is capped at the
    machine's detected physical CPU-core count.
    """
    if started_total is None:
        started_total = time.perf_counter()
    report.mode = "literal"
    report.literal_batch_size = LITERAL_SEARCH_FILES_PER_WORKER_BATCH
    effective_case_flags = normalize_case_flags(case_flags, len(query_terms))

    _notify_progress(progress_callback, "Loading document list", 0.05)
    document_list_started = time.perf_counter()
    if selected_doc_ids is None:
        document_ids = sorted(all_document_ids(index))
        report.selection_applied = False
    else:
        report.selection_applied = True
        report.selected_doc_id_count = len(selected_doc_ids)
        document_ids = sorted({int(doc_id) for doc_id in selected_doc_ids})
    report.literal_document_list_ms = (
        time.perf_counter() - document_list_started
    ) * 1000.0

    report.candidate_docs_after_fts = len(document_ids)
    report.candidate_docs_before_selection = len(document_ids)
    report.candidate_docs_after_selection = len(document_ids)
    report.candidate_docs_before_positions = len(document_ids)

    if not document_ids:
        report.diagnostics.append("No indexed documents are selected.")
        report.total_ms = (time.perf_counter() - started_total) * 1000.0
        _notify_progress(progress_callback, "No indexed documents selected", 1.0)
        return [], report

    # Keep only match/result structures for matching documents. Source text is
    # retained only inside each active worker's complete 100-document batch.
    result_payloads: dict[int, tuple[
        tuple[int, int, int],
        list[list[dict]],
        list[list[dict]],
        list[list[dict]],
        dict,
    ]] = {}

    query_prepare_started = time.perf_counter()
    stripped_queries = [query.strip() for query in query_terms]
    # UTF-8 encoding is performed once per query. Case-insensitive folding
    # handles ASCII A-Z plus German Ä/ä, Ö/ö, Ü/ü and ẞ/ß directly in bytes.
    # No Unicode decoding, regex fallback, normalization or ß-to-ss mapping
    # is used during scanning.
    literal_query_bytes: list[bytes] = []
    literal_query_bytes_lower: list[bytes] = []
    literal_query_needs_case_fold: list[bool] = []
    for query in stripped_queries:
        try:
            encoded = query.encode("utf-8") if query else b""
        except UnicodeEncodeError as exc:
            raise ValueError(
                "Literal search queries must contain valid Unicode text."
            ) from exc
        literal_query_bytes.append(encoded)
        literal_query_bytes_lower.append(
            literal_english_german_lower_bytes(encoded)
        )
        literal_query_needs_case_fold.append(
            literal_query_needs_case_folding(encoded)
        )
    report.literal_query_prepare_ms = (
        time.perf_counter() - query_prepare_started
    ) * 1000.0
    report.literal_raw_term_stats = [
        {
            "query_index": index + 1,
            "query": query,
            "documents_checked": 0,
            "documents_with_matches": 0,
            "occurrences": 0,
        }
        for index, query in enumerate(stripped_queries)
    ]

    def record_raw_term_matches(
        worker_report: SearchReport,
        query_index: int,
        matches: MatchArrays,
    ) -> None:
        """Record pre-proximity raw occurrences for one query/document pair."""
        count_started = time.perf_counter()
        stats = worker_report.literal_raw_term_stats[query_index]
        occurrence_count = int(matches.starts.size)
        stats["documents_checked"] += 1
        stats["documents_with_matches"] += int(occurrence_count > 0)
        stats["occurrences"] += occurrence_count
        worker_report.literal_raw_occurrences_total += occurrence_count
        worker_report.literal_raw_count_ms += (
            time.perf_counter() - count_started
        ) * 1000.0

    def record_raw_candidate_document(
        worker_report: SearchReport,
        doc_id: int,
        document: dict,
        match_arrays: Sequence[MatchArrays],
    ) -> None:
        """Record documents containing every query term before proximity."""
        count_started = time.perf_counter()
        occurrence_counts = [int(matches.starts.size) for matches in match_arrays]
        if not occurrence_counts or any(count == 0 for count in occurrence_counts):
            worker_report.literal_raw_count_ms += (
                time.perf_counter() - count_started
            ) * 1000.0
            return

        total_occurrences = sum(occurrence_counts)
        worker_report.literal_raw_candidate_docs += 1
        worker_report.literal_raw_candidate_occurrences += total_occurrences
        if total_occurrences > worker_report.literal_raw_candidate_max_occurrences:
            worker_report.literal_raw_candidate_max_occurrences = total_occurrences
            worker_report.literal_raw_candidate_max_doc_id = int(doc_id)

        raw_path = str(document.get("path") or document.get("name") or doc_id)
        worker_report.literal_raw_candidate_top_docs.append(
            {
                "doc_id": int(doc_id),
                "path": raw_path,
                "occurrences": total_occurrences,
                "per_term": occurrence_counts,
            }
        )
        worker_report.literal_raw_candidate_top_docs.sort(
            key=lambda item: (-item["occurrences"], item["doc_id"])
        )
        del worker_report.literal_raw_candidate_top_docs[10:]
        worker_report.literal_raw_count_ms += (
            time.perf_counter() - count_started
        ) * 1000.0

    report_fields_to_merge = (
        "source_text_ms",
        "source_text_docs",
        "match_build_ms",
        "proximity_ms",
        "proximity_docs_checked",
        "proximity_docs_with_hits",
        "result_build_ms",
        "runtime_source_file_reads",
        "runtime_source_bytes_read",
        "runtime_lancedb_document_fallbacks",
        "literal_byte_normalize_ms",
        "literal_byte_match_ms",
        "literal_decode_ms",
        "literal_bytes_searched",
        "literal_byte_search_docs",
        "literal_decoded_docs",
        "literal_thread_cpu_ms",
        "literal_raw_occurrences_total",
        "literal_raw_count_ms",
        "literal_raw_candidate_docs",
        "literal_raw_candidate_occurrences",
    )

    def process_document_batch(
        batch_number: int,
        batch_ids: list[int],
    ) -> tuple[dict[int, tuple], SearchReport]:
        """Load one complete batch, byte-scan it, then decode only candidates."""
        batch_started = time.perf_counter()
        thread_cpu_started = time.thread_time()
        thread_name = threading.current_thread().name
        worker_report = SearchReport(mode="literal")
        worker_report.source_text_docs = 0
        worker_report.literal_raw_term_stats = [
            {
                "query_index": index + 1,
                "query": query,
                "documents_checked": 0,
                "documents_with_matches": 0,
                "occurrences": 0,
            }
            for index, query in enumerate(stripped_queries)
        ]
        # Context variables do not automatically propagate to worker threads.
        # Set a worker-local report so rare LanceDB fallbacks are attributed
        # to this batch rather than the GUI thread.
        report_token = _ACTIVE_DB_QUERY_REPORT.set(worker_report)
        try:
            source_started = time.perf_counter()
            source_documents = fetch_literal_search_documents(
                index, batch_ids, worker_report
            )
            batch_load_ms = (time.perf_counter() - source_started) * 1000.0
            worker_report.source_text_ms += batch_load_ms
            worker_report.source_text_docs += len(source_documents)

            batch_payloads: dict[int, tuple] = {}
            processing_started = time.perf_counter()
            for doc_id in batch_ids:
                document = source_documents.get(int(doc_id))
                if document is None:
                    continue

                raw_content = document.get("raw_content")
                if isinstance(raw_content, bytes):
                    scan_bytes = raw_content
                else:
                    # Rare LanceDB fallback documents arrive as Python strings.
                    # Encode them to the same UTF-8 byte representation and use
                    # exactly the same byte-only matching path.
                    scan_bytes = str(document.get("content", "")).encode("utf-8")
                    raw_content = scan_bytes

                match_pairs: list[tuple[MatchArrays, bool]] = []
                lowered_content: bytes | None = None
                lowered_sharp_s_positions: tuple[int, ...] = ()
                document_used_byte_search = False

                for query_index, (
                    query,
                    sensitive,
                    encoded_query,
                    lowered_query,
                    needs_case_fold,
                ) in enumerate(zip(
                    stripped_queries,
                    effective_case_flags,
                    literal_query_bytes,
                    literal_query_bytes_lower,
                    literal_query_needs_case_fold,
                )):
                    if not query:
                        matches = MatchArrays.empty()
                    else:
                        if sensitive or not needs_case_fold:
                            byte_query = encoded_query
                            byte_haystack = scan_bytes
                        else:
                            if lowered_content is None:
                                normalize_started = time.perf_counter()
                                (
                                    lowered_content,
                                    lowered_sharp_s_positions,
                                ) = literal_english_german_lower_bytes_with_offsets(
                                    scan_bytes
                                )
                                worker_report.literal_byte_normalize_ms += (
                                    time.perf_counter() - normalize_started
                                ) * 1000.0
                            byte_query = lowered_query
                            byte_haystack = lowered_content

                        byte_search_started = time.perf_counter()
                        matches = find_literal_byte_matches(
                            byte_haystack,
                            byte_query,
                        )
                        if not sensitive and needs_case_fold:
                            restore_literal_byte_match_offsets(
                                matches,
                                lowered_sharp_s_positions,
                            )
                        worker_report.literal_byte_match_ms += (
                            time.perf_counter() - byte_search_started
                        ) * 1000.0
                        worker_report.literal_bytes_searched += len(scan_bytes)
                        document_used_byte_search = True

                    record_raw_term_matches(worker_report, query_index, matches)
                    # Every literal term has UTF-8 byte offsets at this point.
                    match_pairs.append((matches, True))

                if document_used_byte_search:
                    worker_report.literal_byte_search_docs += 1

                # No Unicode fallback is permitted. If any term has no exact
                # UTF-8 byte occurrence (with ASCII-only folding when enabled),
                # this document cannot satisfy the complete literal query.
                if any(len(matches) == 0 for matches, _uses_bytes in match_pairs):
                    continue

                # Decode only raw candidate documents. Conversion to character
                # offsets preserves the existing character-distance semantics
                # for proximity checks and the positions expected by snippets.
                decode_started = time.perf_counter()
                content = decode_literal_content_and_convert_byte_positions(
                    scan_bytes,
                    match_pairs,
                )
                worker_report.literal_decode_ms += (
                    time.perf_counter() - decode_started
                ) * 1000.0
                worker_report.literal_decoded_docs += 1
                match_arrays = [matches for matches, _uses_bytes in match_pairs]
                document["content"] = content
                document.pop("raw_content", None)

                # This is the raw candidate count: every query component has
                # at least one literal occurrence, before distance/proximity
                # constraints remove documents.
                record_raw_candidate_document(
                    worker_report,
                    int(doc_id),
                    document,
                    match_arrays,
                )

                proximity_started = time.perf_counter()
                chains = find_proximity_chains_numpy(
                    match_arrays,
                    query_terms,
                    index.id_to_term,
                    config,
                    case_sensitive=effective_case_flags,
                    literal_english_german_casefold=True,
                )
                worker_report.proximity_ms += (
                    time.perf_counter() - proximity_started
                ) * 1000.0
                worker_report.proximity_docs_checked += 1
                if not chains:
                    continue

                worker_report.proximity_docs_with_hits += 1
                ranked_chains, score = rank_and_limit_chains(
                    chains,
                    config.effective_max_results_per_document,
                )
                output_started = time.perf_counter()
                for chain in ranked_chains:
                    for match in chain:
                        match["indexed_term"] = str(match.get("query_term", ""))
                public_matches = [
                    build_matches_for_document(
                        query, matches, index.id_to_term, "regex"
                    )
                    for query, matches in zip(query_terms, match_arrays)
                ]
                for term_matches in public_matches:
                    for match in term_matches:
                        match["indexed_term"] = str(match.get("query_term", ""))

                wrapper_annotations: list[list[dict]] = []
                for term_matches in public_matches:
                    slot_annotations: list[dict] = []
                    for match in term_matches:
                        component_annotations = match.get("component_annotations") or []
                        if component_annotations:
                            slot_annotations.extend(
                                {
                                    "query_term": str(
                                        annotation.get(
                                            "query_term", match.get("query_term", "")
                                        )
                                    ),
                                    "start": int(annotation.get("start", 0)),
                                    "end": int(annotation.get("end", 0)),
                                    "wrapper_flag": str(
                                        annotation.get("wrapper_flag", "")
                                    ),
                                }
                                for annotation in component_annotations
                            )
                        else:
                            slot_annotations.append(
                                {
                                    "query_term": str(match.get("query_term", "")),
                                    "start": int(match["start"]),
                                    "end": int(match["end"]),
                                    "wrapper_flag": str(match.get("wrapper_flag", "")),
                                }
                            )
                    wrapper_annotations.append(slot_annotations)

                worker_report.result_build_ms += (
                    time.perf_counter() - output_started
                ) * 1000.0
                batch_payloads[int(doc_id)] = (
                    score,
                    public_matches,
                    ranked_chains,
                    wrapper_annotations,
                    document,
                )

            batch_processing_ms = (
                time.perf_counter() - processing_started
            ) * 1000.0
            thread_cpu_ms = (time.thread_time() - thread_cpu_started) * 1000.0
            worker_report.literal_thread_cpu_ms += thread_cpu_ms
            worker_report.match_build_ms += max(
                0.0,
                batch_processing_ms
                - worker_report.proximity_ms
                - worker_report.result_build_ms,
            )
            worker_report.literal_worker_batches.append(
                {
                    "batch_number": int(batch_number),
                    "thread_name": thread_name,
                    "files_assigned": len(batch_ids),
                    "files_loaded": len(source_documents),
                    "matching_documents": len(batch_payloads),
                    "raw_candidate_documents": worker_report.literal_raw_candidate_docs,
                    "raw_candidate_occurrences": worker_report.literal_raw_candidate_occurrences,
                    "raw_count_ms": worker_report.literal_raw_count_ms,
                    "load_ms": batch_load_ms,
                    "processing_ms": batch_processing_ms,
                    "match_ms": worker_report.match_build_ms,
                    "byte_normalize_ms": worker_report.literal_byte_normalize_ms,
                    "byte_match_ms": worker_report.literal_byte_match_ms,
                    "decode_ms": worker_report.literal_decode_ms,
                    "proximity_ms": worker_report.proximity_ms,
                    "result_build_ms": worker_report.result_build_ms,
                    "bytes_searched": worker_report.literal_bytes_searched,
                    "thread_cpu_ms": thread_cpu_ms,
                    "batch_elapsed_ms": (
                        time.perf_counter() - batch_started
                    ) * 1000.0,
                }
            )
            return batch_payloads, worker_report
        finally:
            _ACTIVE_DB_QUERY_REPORT.reset(report_token)

    total_documents = len(document_ids)
    batches = [
        document_ids[start:start + LITERAL_SEARCH_FILES_PER_WORKER_BATCH]
        for start in range(0, total_documents, LITERAL_SEARCH_FILES_PER_WORKER_BATCH)
    ]
    worker_count = min(get_physical_core_count(), len(batches))
    report.literal_batch_count = len(batches)
    report.literal_worker_count = worker_count
    completed_documents = 0

    # Each active worker loads its entire 100-file batch before matching. No
    # worker shares or mutates result dictionaries; aggregation happens here
    # in the caller thread. Thus peak source-text memory is bounded to roughly
    # 100 files per physical core.
    parallel_started = time.perf_counter()
    parallel_process_cpu_started = time.process_time()
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="literal-search",
    ) as executor:
        futures = {
            executor.submit(process_document_batch, batch_number, batch_ids): batch_ids
            for batch_number, batch_ids in enumerate(batches, start=1)
        }
        for future in concurrent.futures.as_completed(futures):
            batch_ids = futures[future]
            batch_payloads, worker_report = future.result()
            result_payloads.update(batch_payloads)
            report.literal_worker_batches.extend(
                worker_report.literal_worker_batches
            )
            for field_name in report_fields_to_merge:
                setattr(
                    report,
                    field_name,
                    getattr(report, field_name) + getattr(worker_report, field_name),
                )

            for query_index, worker_stats in enumerate(
                worker_report.literal_raw_term_stats
            ):
                aggregate_stats = report.literal_raw_term_stats[query_index]
                for counter_name in (
                    "documents_checked",
                    "documents_with_matches",
                    "occurrences",
                ):
                    aggregate_stats[counter_name] += worker_stats[counter_name]

            if (
                worker_report.literal_raw_candidate_max_occurrences
                > report.literal_raw_candidate_max_occurrences
            ):
                report.literal_raw_candidate_max_occurrences = (
                    worker_report.literal_raw_candidate_max_occurrences
                )
                report.literal_raw_candidate_max_doc_id = (
                    worker_report.literal_raw_candidate_max_doc_id
                )
            report.literal_raw_candidate_top_docs.extend(
                worker_report.literal_raw_candidate_top_docs
            )

            completed_documents += len(batch_ids)
            fraction = min(1.0, completed_documents / total_documents)
            _notify_progress(
                progress_callback,
                (
                    f"Literal text search "
                    f"({completed_documents:,}/{total_documents:,}; "
                    f"{worker_count} workers, "
                    f"{LITERAL_SEARCH_FILES_PER_WORKER_BATCH} files/batch)"
                ),
                0.10 + 0.62 * fraction,
            )
    report.literal_parallel_wall_ms = (
        time.perf_counter() - parallel_started
    ) * 1000.0
    report.literal_parallel_process_cpu_ms = (
        time.process_time() - parallel_process_cpu_started
    ) * 1000.0

    report.candidate_docs_after_postings = len(result_payloads)

    sort_started = time.perf_counter()
    report.literal_docs_before_limit = len(result_payloads)
    scored_hits = [
        (*payload[0], doc_id)
        for doc_id, payload in result_payloads.items()
    ]
    scored_hits.sort(
        key=lambda item: (-item[0], -item[1], -item[2], item[3])
    )
    if max_results is not None:
        scored_hits = scored_hits[:max_results]

    report.literal_docs_after_limit = len(scored_hits)
    hit_ids = [item[3] for item in scored_hits]
    report.literal_sort_limit_ms = (time.perf_counter() - sort_started) * 1000.0
    _notify_progress(progress_callback, "Fetching result documents", 0.88)
    # Matching documents are already present in their worker's loaded batch.
    # Reuse those exact source texts instead of reopening and hashing them.
    report.documents_ms = 0.0

    result_started = time.perf_counter()
    hits: list[SearchHit] = []
    for doc_id in hit_ids:
        (
            _score,
            public_matches,
            ranked_chains,
            wrapper_annotations,
            document,
        ) = result_payloads[doc_id]
        hits.append(
            SearchHit(
                doc_id=doc_id,
                document=document,
                matches_by_term=public_matches,
                chains=ranked_chains,
                mode="literal",
                term_modes=["literal"] * len(query_terms),
                wrapper_annotations=wrapper_annotations,
            )
        )

    report.literal_result_assembly_ms = (
        time.perf_counter() - result_started
    ) * 1000.0
    report.result_build_ms += report.literal_result_assembly_ms
    report.final_documents = len(hits)
    raw_summary_started = time.perf_counter()
    report.literal_raw_candidate_top_docs.sort(
        key=lambda item: (-item["occurrences"], item["doc_id"])
    )
    del report.literal_raw_candidate_top_docs[10:]
    report.literal_raw_summary_ms = (
        time.perf_counter() - raw_summary_started
    ) * 1000.0
    report.total_ms = (time.perf_counter() - started_total) * 1000.0
    report.diagnostics.append(
        f"Literal text search scanned source text in {total_documents:,} selected document(s) "
        f"using {worker_count} worker(s), with {LITERAL_SEARCH_FILES_PER_WORKER_BATCH} "
        "complete files loaded per worker batch; the index vocabulary, postings, regex "
        "syntax, fuzzy matching, wildcard expansion, and Unicode fallback were bypassed. "
        "Case-insensitive matching folds English A-Z and German Ä/ä, Ö/ö, Ü/ü, ẞ/ß directly in UTF-8 bytes; ß and ss remain distinct. "
        f"{report.literal_raw_candidate_docs:,} document(s) contained every query term before "
        f"proximity filtering; {report.proximity_docs_with_hits:,} passed proximity and "
        f"{report.final_documents:,} were returned after result limiting."
    )
    _notify_progress(progress_callback, "Search complete", 1.0)
    return hits, report


# =============================================================================
# Search execution pipeline
# =============================================================================

def search_index(
    index: "SearchIndex",
    terms: list[str],
    mode: TermSearchMode,
    config: SearchConfig,
    max_results: int | None = MAX_RESULTS,
    progress_callback: ProgressCallback | None = None,
    selected_doc_ids: set[int] | None = None,
) -> tuple[list[SearchHit], SearchReport]:
    started_total = time.perf_counter()
    report = SearchReport(mode=str(mode))
    db_report_token = _ACTIVE_DB_QUERY_REPORT.set(report)
    try:
        query_terms = [term.strip() for term in terms if term.strip()]
        case_flags = normalize_case_flags(config.case_sensitive, len(query_terms))
        term_modes = normalize_term_modes(
            config.term_modes or None,
            len(query_terms),
            mode,
        )
        overall_mode = combined_term_mode(term_modes)
        report.mode = overall_mode
        if not query_terms:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            return [], report

        if config.literal_text_search:
            return search_literal_text(
                index,
                query_terms,
                case_flags,
                config,
                max_results,
                report,
                progress_callback=progress_callback,
                selected_doc_ids=selected_doc_ids,
                started_total=started_total,
            )

        selected_ids_filter: set[int] | None = None
        if selected_doc_ids is not None:
            report.selection_applied = True
            selected_ids_filter = {int(doc_id) for doc_id in selected_doc_ids}
            report.selected_doc_id_count = len(selected_ids_filter)
            if not selected_ids_filter:
                report.diagnostics.append("No indexed documents are selected.")
                report.total_ms = (time.perf_counter() - started_total) * 1000.0
                _notify_progress(progress_callback, "No indexed selected documents", 1.0)
                return [], report

        # Expand each whitespace-containing GUI keyword into indexed phrase
        # components. Single-token keyword slots retain the original code path.
        phrase_specs: list[PhraseSpec | None] = []
        slot_component_indices: list[list[int]] = []
        flat_query_terms: list[str] = []
        flat_case_flags: list[bool] = []
        flat_term_modes: list[TermSearchMode] = []
        flat_required_wrappers: list[str | None] = []
        flat_canonical_terms: list[str] = []
        flat_positional_wildcards: list[bool] = []

        for slot, query_term in enumerate(query_terms):
            phrase_spec = parse_phrase_query(query_term)
            if phrase_spec is not None:
                phrase_specs.append(phrase_spec)
                components = phrase_spec.components
            else:
                phrase_specs.append(None)
                components = (
                    PhraseComponent(
                        query_term=query_term,
                        query_start=0,
                        query_end=len(query_term),
                        required_wrapper=query_wrapper_flag(query_term),
                    ),
                )

            component_indices: list[int] = []
            for component in components:
                component_indices.append(len(flat_query_terms))
                flat_query_terms.append(component.query_term)
                flat_case_flags.append(case_flags[slot])
                flat_term_modes.append(term_modes[slot])
                flat_required_wrappers.append(component.required_wrapper)
                flat_canonical_terms.append(canonical_query_term(component.query_term))
                flat_positional_wildcards.append(
                    component.wildcard_pattern is not None
                )
            slot_component_indices.append(component_indices)

        # 1. Resolve each indexed component using the existing normal/additional
        # vocabulary paths. Literal punctuation stays in PhraseSpec separators.
        _notify_progress(progress_callback, "Variant resolution", 0.10)
        variants_by_component, routes = resolve_query_terms(
            index,
            flat_query_terms,
            mode,
            config,
            report,
            term_modes=flat_term_modes,
            positional_wildcards=flat_positional_wildcards,
        )

        unresolved_components = [
            flat_query_terms[index_number]
            for index_number, variants in enumerate(variants_by_component)
            if not variants and not flat_positional_wildcards[index_number]
        ]
        if unresolved_components:
            report.diagnostics.append(
                "No indexed variant for: "
                + ", ".join(repr(term) for term in unresolved_components)
                + ". Database mode does not scan original text."
            )
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            _notify_progress(progress_callback, "No indexed match", 1.0)
            return [], report

        # 2. Candidate documents must contain every indexed component across
        # every keyword slot. Punctuation-only phrase separators are verified
        # later from the stored source text and do not need vocabulary entries.
        _notify_progress(progress_callback, "Candidate lookup", 0.20)
        candidate_docs = fts_candidate_docs(
            index,
            flat_query_terms,
            routes,
            variants_by_component,
            report,
        )
        if candidate_docs is None:
            candidate_docs = all_document_ids(index)
            report.diagnostics.append(
                f"Tantivy unavailable; using all {len(candidate_docs):,} document(s) as candidates."
            )
        report.candidate_docs_after_fts = len(candidate_docs)

        if not candidate_docs:
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            _notify_progress(progress_callback, "No results found", 1.0)
            return [], report

        # Apply the GUI's checked-file selection immediately after candidate
        # lookup. Duplicate physical copies share one doc_id, and the GUI
        # resolver includes that ID whenever any of its copies is selected.
        if selected_ids_filter is not None:
            selection_started = time.perf_counter()
            report.candidate_docs_before_selection = len(candidate_docs)
            candidate_docs.intersection_update(selected_ids_filter)
            report.candidate_docs_after_selection = len(candidate_docs)
            report.selection_filter_ms = (
                time.perf_counter() - selection_started
            ) * 1000.0
            if not candidate_docs:
                report.total_ms = (time.perf_counter() - started_total) * 1000.0
                _notify_progress(progress_callback, "No results in selected files", 1.0)
                return [], report
        else:
            report.candidate_docs_before_selection = len(candidate_docs)
            report.candidate_docs_after_selection = len(candidate_docs)

        # 3. Use per-component frequencies to discard documents missing an
        # indexed phrase component and retain the rarest-token anchor plan.
        _notify_progress(progress_callback, "Frequency planning", 0.30)
        indexed_terms: list[str] = []
        indexed_variants: list[dict[int, int]] = []
        indexed_component_indices: list[int] = []
        for component_index, (query_term, variants, positional_wildcard) in enumerate(zip(
            flat_query_terms,
            variants_by_component,
            flat_positional_wildcards,
        )):
            if variants and not positional_wildcard:
                indexed_terms.append(query_term)
                indexed_variants.append(variants)
                indexed_component_indices.append(component_index)

        frequency_started = time.perf_counter()
        if indexed_terms:
            frequency_by_doc_term, frequency_rows, frequency_pairs = fetch_term_frequencies(
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
            frequency_pairs = set()
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

        # 4. Fetch all required postings once for all flattened components.
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
            required_pairs=frequency_pairs,
        )
        report.postings_ms = (time.perf_counter() - postings_started) * 1000.0
        report.postings_terms = len(required_term_ids)
        report.postings_rows = sum(len(doc_map) for doc_map in postings.values())
        _notify_progress(progress_callback, "Match building", 0.55)

        has_phrase_queries = any(spec is not None for spec in phrase_specs)
        needs_early_source_content = any(flat_case_flags)
        source_documents: dict[int, dict] = {}
        if needs_early_source_content:
            source_text_started = time.perf_counter()
            if has_phrase_queries:
                source_documents = fetch_indexed_document_contents(
                    index,
                    sorted(candidate_docs),
                )
            else:
                source_documents = fetch_documents(
                    index,
                    sorted(candidate_docs),
                    report,
                )
            report.source_text_ms += (
                time.perf_counter() - source_text_started
            ) * 1000.0
            report.source_text_docs += len(source_documents)

        # 5. Build component occurrence arrays. When every anchor maps to one
        # unambiguous indexed component, build those rarest components first.
        # Documents whose actual anchor occurrences fail wrapper/case checks
        # cannot match the query, so their other component arrays are skipped.
        match_build_started = time.perf_counter()
        phrase_source_fetch_inside_match_ms = 0.0
        component_mappings: list[dict[int, MatchArrays] | None] = [
            None for _ in flat_query_terms
        ]
        anchor_component_for_term: dict[str, list[int]] = {}
        for component_index in indexed_component_indices:
            anchor_component_for_term.setdefault(
                flat_query_terms[component_index].casefold(), []
            ).append(component_index)

        anchor_docs_by_component: dict[int, set[int]] = {}
        anchor_gate_enabled = bool(anchors) and all(
            len(anchor_component_for_term.get(str(info["query_term"]), ())) == 1
            for info in anchors.values()
        )
        if anchor_gate_enabled:
            for doc_id, anchor_info in anchors.items():
                normalized_anchor = str(anchor_info["query_term"])
                component_index = anchor_component_for_term[normalized_anchor][0]
                anchor_docs_by_component.setdefault(component_index, set()).add(
                    int(doc_id)
                )

        report.anchor_docs_before_gate = len(candidate_docs)
        anchor_gate_started = time.perf_counter()
        if anchor_gate_enabled:
            anchor_matching_docs: set[int] = set()
            for component_index, anchor_doc_ids in anchor_docs_by_component.items():
                variants = variants_by_component[component_index]
                required_wrapper = flat_required_wrappers[component_index]
                anchor_mapping = build_match_arrays(
                    variants,
                    postings,
                    required_wrapper=required_wrapper,
                    candidate_doc_ids=anchor_doc_ids,
                )
                if flat_case_flags[component_index]:
                    filtered_anchor_mapping: dict[int, MatchArrays] = {}
                    for doc_id, matches in anchor_mapping.items():
                        document = source_documents.get(int(doc_id))
                        if document is None:
                            continue
                        filtered = filter_case_sensitive_matches(
                            matches,
                            str(document.get("content", "")),
                            flat_canonical_terms[component_index],
                            flat_term_modes[component_index],
                            config.fuzzy_distance,
                        )
                        if len(filtered):
                            filtered_anchor_mapping[doc_id] = filtered
                    anchor_mapping = filtered_anchor_mapping
                component_mappings[component_index] = anchor_mapping
                anchor_matching_docs.update(anchor_mapping)
                report.anchor_component_builds += 1
                report.match_build_by_term.append(
                    {
                        "query_term": flat_query_terms[component_index],
                        "term_type": (
                            "additional"
                            if routes[component_index] == "additional"
                            else "normal"
                        ),
                        "path": "terms/postings (rarest anchor first)",
                        "route": routes[component_index],
                        "matching_docs": len(anchor_mapping),
                        "occurrences": sum(
                            len(items) for items in anchor_mapping.values()
                        ),
                        "time_ms": 0.0,
                    }
                )

            candidate_docs.intersection_update(anchor_matching_docs)

        report.anchor_gate_ms = (
            time.perf_counter() - anchor_gate_started
        ) * 1000.0
        report.anchor_docs_after_gate = len(candidate_docs)
        if anchor_gate_enabled and not candidate_docs:
            report.match_build_ms = max(
                0.0,
                (time.perf_counter() - match_build_started) * 1000.0
                - report.anchor_gate_ms,
            )
            report.total_ms = (time.perf_counter() - started_total) * 1000.0
            _notify_progress(progress_callback, "No results after anchor verification", 1.0)
            return [], report

        # Build remaining components only for documents that passed the anchor
        # gate. This is especially useful for wrapper/case-sensitive queries;
        # ordinary searches usually retain all frequency-qualified documents.
        for component_index, (
            query_term,
            route,
            variants,
            case_sensitive,
            term_mode,
            required_wrapper,
            canonical_term,
            is_positional_wildcard,
        ) in enumerate(
            zip(
                flat_query_terms,
                routes,
                variants_by_component,
                flat_case_flags,
                flat_term_modes,
                flat_required_wrappers,
                flat_canonical_terms,
                flat_positional_wildcards,
            )
        ):
            started = time.perf_counter()
            if is_positional_wildcard:
                # Wildcard phrase components are resolved directly from source
                # coordinates by build_ordered_phrase_matches().
                component_mappings[component_index] = None
                report.match_build_by_term.append(
                    {
                        "query_term": query_term,
                        "term_type": "wildcard",
                        "path": "anchor-driven source-position check",
                        "route": "wildcard",
                        "matching_docs": 0,
                        "occurrences": 0,
                        "time_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )
                continue

            # The anchor component was already built and used to restrict the
            # document set above. Documents whose anchor was a different
            # component still need this component's occurrence array.
            if component_mappings[component_index] is not None:
                anchor_doc_ids = anchor_docs_by_component.get(component_index, set())
                remaining_doc_ids = candidate_docs - anchor_doc_ids
                if not remaining_doc_ids:
                    continue

                mapping = build_match_arrays(
                    variants,
                    postings,
                    required_wrapper=required_wrapper,
                    candidate_doc_ids=remaining_doc_ids,
                )
                if case_sensitive:
                    filtered_mapping: dict[int, MatchArrays] = {}
                    for doc_id, matches in mapping.items():
                        document = source_documents.get(int(doc_id))
                        if document is None:
                            continue
                        filtered = filter_case_sensitive_matches(
                            matches,
                            str(document.get("content", "")),
                            canonical_term,
                            term_mode,
                            config.fuzzy_distance,
                        )
                        if len(filtered):
                            filtered_mapping[doc_id] = filtered
                    mapping = filtered_mapping

                component_mappings[component_index].update(mapping)
                report.match_build_by_term.append(
                    {
                        "query_term": query_term,
                        "term_type": (
                            "additional" if route == "additional" else "normal"
                        ),
                        "path": "terms/postings (non-anchor documents)",
                        "route": route,
                        "matching_docs": len(mapping),
                        "occurrences": sum(len(items) for items in mapping.values()),
                        "time_ms": (time.perf_counter() - started) * 1000.0,
                    }
                )
                continue

            if not variants:
                mapping = {}
                path_used = "no indexed variant"
            else:
                mapping = build_match_arrays(
                    variants,
                    postings,
                    required_wrapper=required_wrapper,
                    candidate_doc_ids=candidate_docs,
                )
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
                            canonical_term,
                            term_mode,
                            config.fuzzy_distance,
                        )
                        if len(filtered):
                            filtered_mapping[doc_id] = filtered
                    mapping = filtered_mapping

            component_mappings[component_index] = mapping
            elapsed = (time.perf_counter() - started) * 1000.0
            report.match_build_by_term.append(
                {
                    "query_term": query_term,
                    "term_type": (
                        "additional" if route == "additional"
                        else "unindexed" if route == "unindexed"
                        else "normal"
                    ),
                    "path": path_used,
                    "route": route,
                    "matching_docs": len(mapping),
                    "occurrences": sum(len(items) for items in mapping.values()),
                    "time_ms": elapsed,
                }
            )

        report.match_build_ms = max(
            0.0,
            (time.perf_counter() - match_build_started) * 1000.0
            - report.anchor_gate_ms
            - phrase_source_fetch_inside_match_ms,
        )

        # 6. First apply a safe positional geometry filter. Exact whitespace and
        # punctuation, wildcard contents, and phrase-edge boundaries are still
        # checked later against the indexed text snapshot. This narrows the set
        # of full-text rows that need to be transferred from LanceDB.
        phrase_docs_by_slot: dict[int, set[int]] = {}
        phrase_prefilter_started = time.perf_counter()
        if has_phrase_queries:
            for slot, (phrase_spec, component_indices) in enumerate(
                zip(phrase_specs, slot_component_indices)
            ):
                if phrase_spec is None:
                    continue
                phrase_component_mappings = [
                    component_mappings[index] for index in component_indices
                ]
                report.phrase_docs_before_prefilter += len(candidate_docs)
                phrase_docs = phrase_position_candidate_docs(
                    phrase_spec,
                    phrase_component_mappings,
                    candidate_docs,
                )
                phrase_docs_by_slot[slot] = phrase_docs
                report.phrase_docs_after_prefilter += len(phrase_docs)
        report.phrase_prefilter_ms = (
            time.perf_counter() - phrase_prefilter_started
        ) * 1000.0

        if has_phrase_queries and not needs_early_source_content:
            phrase_docs_to_fetch = (
                set().union(*phrase_docs_by_slot.values())
                if phrase_docs_by_slot
                else set()
            )
            if phrase_docs_to_fetch:
                _notify_progress(
                    progress_callback,
                    "Loading phrase verification text",
                    0.64,
                )
                source_text_started = time.perf_counter()
                source_documents = fetch_indexed_document_contents(
                    index,
                    sorted(phrase_docs_to_fetch),
                )
                fetch_elapsed = (time.perf_counter() - source_text_started) * 1000.0
                report.source_text_ms += fetch_elapsed
                report.source_text_docs += len(source_documents)
                phrase_source_fetch_inside_match_ms += fetch_elapsed

        # 7. Recombine each phrase's token arrays into one logical MatchArrays
        # slot. Only geometrically possible phrase documents are loaded and
        # passed to the exact source-boundary verifier.
        match_arrays_by_term: list[dict[int, MatchArrays]] = []
        for slot, (query_term, component_indices, phrase_spec) in enumerate(
            zip(query_terms, slot_component_indices, phrase_specs)
        ):
            if phrase_spec is None:
                mapping = component_mappings[component_indices[0]]
            else:
                phrase_started = time.perf_counter()
                phrase_component_mappings = [
                    component_mappings[index] for index in component_indices
                ]
                phrase_candidate_docs = phrase_docs_by_slot.get(slot, set())
                mapping = build_ordered_phrase_matches(
                    phrase_spec,
                    phrase_component_mappings,
                    source_documents,
                    phrase_candidate_docs,
                )
                report.match_build_by_term.append(
                    {
                        "query_term": query_term,
                        "term_type": "phrase",
                        "path": (
                            "indexed ordered phrase "
                            "(positional prefilter + exact boundary check)"
                        ),
                        "route": "phrase",
                        "matching_docs": len(mapping),
                        "occurrences": sum(len(items) for items in mapping.values()),
                        "time_ms": (time.perf_counter() - phrase_started) * 1000.0,
                    }
                )
            match_arrays_by_term.append(mapping)

        report.match_build_ms = max(
            0.0,
            (time.perf_counter() - match_build_started) * 1000.0
            - phrase_source_fetch_inside_match_ms,
        )

        # 7. Authoritative document intersection at the original GUI keyword
        # slot level: a phrase counts as one keyword, not N separate keywords.
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

        # 8. Existing NumPy proximity engine works on original keyword slots.
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
            # The posting-level anchor may be an inner phrase component. The
            # composite phrase slot is therefore allowed to select its own
            # rarest occurrence anchor inside find_proximity_chains_numpy.
            anchor_term = (
                anchor_info["query_term"]
                if anchor_info and anchor_info["query_term"] in {
                    query_term.casefold() for query_term in query_terms
                }
                else None
            )
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
                    term_mode,
                )
                for query_term, mapping, term_mode in zip(
                    query_terms,
                    match_arrays_by_term,
                    term_modes,
                )
            ]
            public_wrapper_annotations: list[list[dict]] = []
            for term_matches in public_matches:
                slot_annotations: list[dict] = []
                for match in term_matches:
                    component_annotations = match.get("component_annotations") or []
                    if component_annotations:
                        for annotation in component_annotations:
                            slot_annotations.append(
                                {
                                    "query_term": str(annotation.get("query_term", match.get("query_term", ""))),
                                    "start": int(annotation.get("start", 0)),
                                    "end": int(annotation.get("end", 0)),
                                    "wrapper_flag": str(annotation.get("wrapper_flag", "")),
                                }
                            )
                    else:
                        slot_annotations.append(
                            {
                                "query_term": str(match.get("query_term", "")),
                                "start": int(match["start"]),
                                "end": int(match["end"]),
                                "wrapper_flag": str(match.get("wrapper_flag", "")),
                            }
                        )
                public_wrapper_annotations.append(slot_annotations)

            hits.append(
                SearchHit(
                    doc_id=doc_id,
                    document=documents.get(doc_id, {}),
                    matches_by_term=public_matches,
                    chains=hit_chains[doc_id],
                    mode=overall_mode,
                    term_modes=list(term_modes),
                    wrapper_annotations=public_wrapper_annotations,
                )
            )

        report.result_build_ms = (
            time.perf_counter() - result_started
        ) * 1000.0
        report.final_documents = len(hits)
        report.total_ms = (time.perf_counter() - started_total) * 1000.0

        if not hits:
            report.diagnostics.append("no results found")
            _notify_progress(progress_callback, "No results found", 1.0)
        else:
            _notify_progress(progress_callback, "Search complete", 1.0)
        return hits, report
    finally:
        _ACTIVE_DB_QUERY_REPORT.reset(db_report_token)

# =============================================================================
# Document deletion and path collision handling
# =============================================================================

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
    # Deletion can remove a doc_id's final physical location and its postings.
    # Invalidate positional cache entries before mutating the tables.
    index.posting_cache.clear()
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

# =============================================================================
# Public engine facade
# =============================================================================

class TextSearchEngine:
    # Startup, state, and database lifecycle
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
        self._folder_snapshot: dict[str, tuple[int, int, int, int, int]] = (
            self._load_source_manifest()
        )
        _notify_progress(progress_callback, "Opening database", 0.05)
        self.db = lancedb.connect(str(self.db_path))
        self._maintenance_modifications, previous_startup_count = self._load_maintenance_state()
        self._startup_count = previous_startup_count + 1
        self._save_maintenance_modifications()

        # Open the persistent tables first, then reconcile source files below.
        # Obsolete-version cleanup still runs only once every 20th startup and
        # is never triggered by ordinary indexing operations.
        _notify_progress(progress_callback, "Checking database schema", 0.12)
        self._ensure_empty_schema_if_missing(progress_callback)
        self.index: SearchIndex | None = self._open_if_valid(
            progress_callback=progress_callback,
        )

        # Reconcile source files on startup. A persistent stat manifest avoids
        # hashing unchanged paths; new/changed paths are read and hashed once.
        # A complete SHA-256 pass is forced periodically as a safety check.
        try:
            _notify_progress(progress_callback, "Synchronizing text folder", 0.82)
            startup_sync = self.sync_search_folder(
                progress_callback=progress_callback,
                _startup_check=True,
            )
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
    def _source_manifest_path(self) -> Path:
        return self.db_path / SOURCE_MANIFEST_FILE

    def _load_source_manifest(self) -> dict[str, tuple[int, int, int, int, int]]:
        """Load prior file fingerprints used to avoid hashing unchanged files."""
        try:
            payload = json.loads(
                self._source_manifest_path.read_text(encoding="utf-8")
            )
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read source-file manifest: %s", exc)
            return {}

        if not isinstance(payload, dict) or payload.get("version") != SOURCE_MANIFEST_VERSION:
            logger.info("Ignoring missing or unsupported source-file manifest")
            return {}

        files = payload.get("files")
        if not isinstance(files, dict):
            logger.warning("Ignoring malformed source-file manifest: files must be an object")
            return {}

        snapshot: dict[str, tuple[int, int, int, int, int]] = {}
        for path_str, values in files.items():
            if not isinstance(path_str, str) or not isinstance(values, list) or len(values) != 5:
                continue
            try:
                snapshot[path_str] = tuple(int(value) for value in values)
            except (TypeError, ValueError):
                continue
        return snapshot

    def _save_source_manifest(
        self,
        snapshot: dict[str, tuple[int, int, int, int, int]],
    ) -> None:
        """Atomically persist file fingerprints after a successful synchronization."""
        manifest_path = self._source_manifest_path
        temporary_path = manifest_path.with_name(manifest_path.name + ".tmp")
        payload = {
            "version": SOURCE_MANIFEST_VERSION,
            "files": {
                path_str: list(metadata)
                for path_str, metadata in snapshot.items()
            },
        }
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary_path, manifest_path)
        except OSError:
            logger.warning(
                "Could not persist source-file manifest; next startup will verify files again",
                exc_info=True,
            )
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

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
        deleted_folder = (self.search_folder.parent / "deleted").absolute()
        # Include all managed files; a failed recursive scan must abort migration.
        current, snapshot = scan_search_folder(self.search_folder, deleted_folder)
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
        # The just-built index already contains the hashes of these exact files.
        # Publishing their stat fingerprints prevents the immediately following
        # constructor sync from reading and hashing every source a second time.
        self._folder_snapshot = snapshot
        self._save_source_manifest(snapshot)

    def _load_maintenance_state(self) -> tuple[int, int]:
        """Load both maintenance counters from one consistent state-file read."""
        try:
            payload = json.loads(self._maintenance_state_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            # First launch: a missing state file is expected.
            return 0, 0
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read LanceDB maintenance state: %s", exc)
            return 0, 0

        if not isinstance(payload, dict):
            logger.warning("Ignoring malformed LanceDB maintenance state: expected an object")
            return 0, 0

        def read_counter(key: str) -> int:
            try:
                return max(0, int(payload.get(key, 0)))
            except (TypeError, ValueError, OverflowError):
                logger.warning("Invalid %s value in LanceDB maintenance state", key)
                return 0

        return read_counter("logical_modifications"), read_counter("startup_count")

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

    # Database maintenance policy
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
                logger.warning(
                    "Could not check whether the document table is empty before maintenance",
                    exc_info=True,
                )

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

    # Open and activate persisted index state
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

        # Older databases may predate these performance indexes. Creating
        # them is additive and leaves the existing table schema unchanged.
        ensure_search_scalar_indexes(postings, frequencies)

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

    def resolve_paths_to_doc_ids(
        self,
        paths: Sequence[str | Path],
    ) -> tuple[set[int], set[Path]]:
        """Resolve selected physical paths through the in-memory location cache.

        Identical files intentionally share one indexed doc_id. Selecting any
        physical copy therefore selects the shared content document. The
        second return value contains paths that are not currently indexed.
        This avoids querying all database locations on every GUI search.
        """
        with self._index_lock:
            if self.index is None:
                return set(), {Path(path).absolute() for path in paths}

            selected_by_key = {
                normalized_path_key(path): Path(path).absolute()
                for path in paths
            }
            if not selected_by_key:
                return set(), set()

            selected_doc_ids: set[int] = set()
            matched_keys: set[str] = set()
            for raw_doc_id, metadata in self.index.document_metadata.items():
                locations = metadata.get("locations") or [metadata.get("path", "")]
                for location in locations:
                    if not location:
                        continue
                    key = normalized_path_key(str(location))
                    if key in selected_by_key:
                        selected_doc_ids.add(int(raw_doc_id))
                        matched_keys.add(key)

            unindexed_paths = {
                path
                for key, path in selected_by_key.items()
                if key not in matched_keys
            }
            return selected_doc_ids, unindexed_paths

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
        deleted_folder = (self.search_folder.parent / "deleted").absolute()
        # Use the same complete, fail-closed scan as normal synchronization. This
        # also handles .TXT on case-sensitive filesystems and never mistakes a
        # traversal error for an empty folder.
        folder_sources, _snapshot = scan_search_folder(self.search_folder, deleted_folder)

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

    def _activate_rebuilt_index(self, rebuilt) -> None:
        """Attach fresh table handles and rebuild in-memory search caches."""
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
        ) = rebuilt

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

    def _rebuild_and_activate(
        self,
        paths: Sequence[Path],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Bulk-build from a known complete path set and install fresh handles."""
        # Drop the engine's old table references before rebuild_index replaces tables.
        self.index = None
        try:
            rebuilt = rebuild_index(
                self.db,
                collect_paths(list(paths)),
                progress_callback=progress_callback,
            )
            self._activate_rebuilt_index(rebuilt)
        except Exception:
            # If the rebuild failed before/after table replacement, recover usable
            # handles whenever the persistent schema remains valid, then re-raise.
            self.index = None
            try:
                if index_schema_is_valid(self.db):
                    self.index = self._open_if_valid()
            except Exception:
                logger.exception("Could not reopen index after failed bulk rebuild")
            raise

        self._maintenance_modifications = 0
        self._save_maintenance_modifications()

    def _bulk_build_result(self) -> dict[str, int]:
        """Return GUI-compatible counters for a clean bulk build."""
        if self.index is None:
            raise RuntimeError("Bulk build finished without an active index.")
        unique_count = int(self.index.docs.count_rows())
        location_count = int(self.index.locations.count_rows())
        return {
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

    def _ensure_ready(
        self,
        extra_paths: list[Path] | None = None,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        if self.index is not None and index_schema_is_valid(self.db):
            return

        # Gather external locations while the old index handles are still live.
        sources = self._rebuild_sources(extra_paths or [])
        self._rebuild_and_activate(sources, progress_callback=progress_callback)

    # Explicit indexing and incremental folder synchronization
    def index_files(
        self,
        paths: list[str | Path],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        with self._index_lock:
            deleted_folder = (self.search_folder.parent / "deleted").absolute()
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

            had_valid_index = self.index is not None and index_schema_is_valid(self.db)
            if not had_valid_index:
                # _ensure_ready performs a complete bulk rebuild from the folder
                # plus these explicit paths. Do not immediately re-hash that same
                # content through sync_index after the rebuild.
                self._ensure_ready(normalized, progress_callback=progress_callback)
                return self._bulk_build_result()

            assert self.index is not None
            if int(self.index.docs.count_rows()) == 0:
                # A valid schema with no documents is equivalent to a missing
                # database for initial indexing. Include all current folder files
                # as well as explicitly selected/external paths, exactly as the
                # invalid-schema path does.
                sources = self._rebuild_sources(normalized)
                self._rebuild_and_activate(sources, progress_callback=progress_callback)
                return self._bulk_build_result()

            # Existing non-empty indexes remain incremental: only unseen content
            # hashes are tokenized, and unchanged hashes reuse existing postings.
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
        snapshot: dict[str, tuple[int, int, int, int, int]],
        *,
        trusted_unchanged_paths: set[str] | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, int]:
        """Reconcile a scanned folder and publish its metadata snapshot.

        Only paths whose persisted stat fingerprint changed are SHA-256 checked.
        A complete verification is forced periodically to detect metadata-preserving
        external edits; successful snapshots are atomically persisted for next launch.
        """
        current_strings = {str(path) for path in current}

        if self.index is None or not index_schema_is_valid(self.db):
            self._ensure_ready(current, progress_callback=progress_callback)
            result = self._bulk_build_result()
        else:
            current_document_count = int(self.index.docs.count_rows())
            if current and current_document_count == 0:
                logger.info(
                    "Using bulk index build for %d source file(s): database has zero documents",
                    len(current),
                )
                self._rebuild_and_activate(current, progress_callback=progress_callback)
                result = self._bulk_build_result()
            else:
                old = query_rows(self.index.locations, columns=["path"])
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
        self._save_source_manifest(self._folder_snapshot)
        return result

    def sync_search_folder(
        self,
        *,
        progress_callback: ProgressCallback | None = None,
        _startup_check: bool = False,
    ) -> dict[str, int]:
        """Synchronize the folder using persisted stat fingerprints where possible.

        New or changed paths are read and hashed; unchanged paths reuse their
        database SHA-256. A periodic full verification protects against unusual
        metadata-preserving edits. Background monitoring and ordinary searches do
        not invoke this method.
        """
        with self._index_lock:
            deleted_folder = self.search_folder.parent / "deleted"
            current, snapshot = scan_search_folder(self.search_folder, deleted_folder)
            force_full_verification = (
                _startup_check
                and self._startup_count > 0
                and self._startup_count % SOURCE_MANIFEST_FULL_VERIFY_EVERY_STARTS == 0
            )
            trusted_unchanged_paths = set()
            if not force_full_verification:
                trusted_unchanged_paths = {
                    path_str
                    for path_str, metadata in snapshot.items()
                    if self._folder_snapshot.get(path_str) == metadata
                }
            else:
                logger.info(
                    "Performing periodic full SHA-256 verification on startup %s",
                    self._startup_count,
                )
            return self._sync_scanned_folder(
                current,
                snapshot,
                trusted_unchanged_paths=trusted_unchanged_paths,
                progress_callback=progress_callback,
            )

    # Database reset and source-file disposition
    def reset_database(
        self,
        action: Literal["delete", "move"],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> dict[str, object]:
        """Clear the index safely, then delete or move its source files.

        This deliberately does NOT drop/recreate LanceDB tables. On Windows,
        dropping tables while native LanceDB/Arrow handles are alive can terminate
        the process. Deleting rows in-place preserves the table handles and schema.
        Source files are only changed after the database reset succeeds.
        """
        debug_path = self.db_path.absolute().parent / "reset_debug.log"

        def debug(stage: str, **details: object) -> None:
            try:
                suffix = " | " + ", ".join(
                    f"{key}={value!r}" for key, value in details.items()
                ) if details else ""
                line = (
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                    f"[RESET] thread={threading.current_thread().name} "
                    f"{stage}{suffix}\n"
                )
                with debug_path.open("a", encoding="utf-8", buffering=1) as stream:
                    stream.write(line)
                logger.info("%s", line.rstrip())
            except Exception:
                # Diagnostics must never be the reason reset fails.
                logger.error("Could not write database-reset diagnostics to %s", debug_path, exc_info=True)

        with self._index_lock:
            debug("BEGIN", action=action, db_path=str(self.db_path.absolute()),
                  search_folder=str(self.search_folder))
            if action not in {"delete", "move"}:
                debug("INVALID ACTION", action=action)
                raise ValueError("Unknown database reset action.")

            search_folder = self.search_folder.absolute()
            deleted_folder = search_folder.parent / "deleted"

            # Scan first, and abort before mutating anything if the directory scan
            # is incomplete. Include still-existing indexed source files outside
            # the managed folder as the previous reset behavior did.
            _notify_progress(progress_callback, "Scanning files", 0.05)
            paths: set[Path] = set()
            try:
                scanned_paths, _snapshot = scan_search_folder(search_folder, deleted_folder)
                paths.update(path.absolute() for path in scanned_paths)
                if self.index is not None:
                    for row in query_rows(
                        self.index.locations, columns=["path"],
                        operation="Reset: read indexed source locations",
                    ):
                        path = Path(str(row["path"])).absolute()
                        if (
                            path.suffix.casefold() == ".txt"
                            and path.is_file()
                            and not path_in_folder(path, deleted_folder)
                        ):
                            paths.add(path)
            except Exception:
                debug("SCAN FAILED")
                logger.exception("Database reset failed during source-file scan")
                raise

            ordered_paths = sorted(
                (path for path in paths if not path_in_folder(path, deleted_folder)),
                key=lambda path: str(path).casefold(),
            )
            debug("SCAN COMPLETE", file_count=len(ordered_paths))

            # Keep the existing tables and native handles. Every key column in
            # this index is non-negative, so the predicate clears all rows while
            # staying compatible with LanceDB's SQL filter parser.
            if not index_schema_is_valid(self.db):
                debug("INVALID SCHEMA", tables=self.db.list_tables())
                raise RuntimeError(
                    "Cannot reset: the database tables do not match the current schema. "
                    "No source files have been deleted or moved."
                )

            if self.index is None:
                debug("OPENING CURRENT INDEX")
                self.index = self._open_if_valid(progress_callback=progress_callback)
                if self.index is None:
                    raise RuntimeError("Cannot reset: database tables could not be opened.")

            table_handles = {
                DOCS_TABLE: (self.index.docs, "id"),
                DOCUMENT_LOCATIONS_TABLE: (self.index.locations, "doc_id"),
                TERMS_TABLE: (self.index.terms, "term_id"),
                POSTINGS_TABLE: (self.index.postings, "doc_id"),
                TERM_FREQUENCIES_TABLE: (self.index.frequencies, "doc_id"),
                FTS_DOCS_TABLE: (self.index.fts_docs, "doc_id"),
            }
            # Clear dependent/search acceleration tables first, then document rows
            # and vocabulary. This avoids dangling rows even if a later table fails.
            clear_order = (
                FTS_DOCS_TABLE,
                TERM_FREQUENCIES_TABLE,
                POSTINGS_TABLE,
                DOCUMENT_LOCATIONS_TABLE,
                DOCS_TABLE,
                TERMS_TABLE,
            )
            def refresh_index_caches() -> None:
                """Refresh Python-side lookup maps without reopening native tables."""
                if self.index is None:
                    return
                (
                    self.index.id_to_term,
                    self.index.term_to_id,
                    self.index.term_lengths,
                    self.index.normal_terms_by_length,
                    self.index.additional_terms_by_length,
                ) = load_terms(self.index.terms)
                self.index.normal_terms_by_reversed_length.clear()
                (
                    self.index.document_metadata,
                    self.index.doc_word_counts,
                ) = build_runtime_search_caches(self.index.docs, self.index.locations)

            debug("CLEAR TABLES START")
            for step, table_name in enumerate(clear_order, start=1):
                table, key_column = table_handles[table_name]
                try:
                    before = int(table.count_rows())
                    debug("TABLE BEFORE CLEAR", table=table_name, rows=before)
                    _notify_progress(
                        progress_callback,
                        f"Clearing database table {step}/{len(clear_order)}: {table_name}",
                        0.10 + 0.48 * (step / len(clear_order)),
                    )
                    if before:
                        table.delete(f"{key_column} >= 0")
                    after = int(table.count_rows())
                    debug("TABLE AFTER CLEAR", table=table_name, rows=after)
                    if after != 0:
                        raise RuntimeError(
                            f"Table {table_name!r} still contains {after} row(s) after reset."
                        )
                except Exception:
                    debug("TABLE CLEAR FAILED", table=table_name)
                    logger.exception("Database reset failed clearing table %s", table_name)
                    # Keep the same native table handles and refresh only the
                    # Python-side maps if the partial reset left the schema usable.
                    try:
                        refresh_index_caches()
                    except Exception:
                        logger.exception("Could not refresh caches after partial reset")
                    raise

            debug("ALL TABLES CLEARED")
            _notify_progress(progress_callback, "Refreshing empty index", 0.62)
            # Do not drop/reopen tables and do not force garbage collection here.
            # Reusing the existing LanceDB handles avoids native destructor paths
            # that can terminate some Windows builds rather than raising Python errors.
            refresh_index_caches()

            self._folder_snapshot = {}
            self._save_source_manifest({})
            self._maintenance_modifications = 0
            self._save_maintenance_modifications()
            debug("EMPTY INDEX READY")

            # Only modify source files after the database is successfully empty.
            moved_count = 0
            deleted_count = 0
            file_errors: list[str] = []
            if ordered_paths:
                for number, source_path in enumerate(ordered_paths, start=1):
                    action_label = "Deleting files" if action == "delete" else "Moving files to deleted"
                    _notify_progress(
                        progress_callback,
                        f"{action_label} ({number}/{len(ordered_paths)})",
                        0.62 + 0.36 * (number / len(ordered_paths)),
                    )
                    try:
                        if not source_path.exists():
                            debug("SOURCE ALREADY MISSING", path=str(source_path))
                            continue
                        if action == "delete":
                            source_path.unlink()
                            deleted_count += 1
                            debug("FILE DELETED", path=str(source_path))
                        else:
                            deleted_folder.mkdir(parents=True, exist_ok=True)
                            destination = _unique_deleted_destination(deleted_folder, source_path)
                            shutil.move(str(source_path), str(destination))
                            moved_count += 1
                            debug("FILE MOVED", source=str(source_path), destination=str(destination))
                    except Exception as exc:
                        details = f"{source_path}: {type(exc).__name__}: {exc}"
                        file_errors.append(details)
                        debug("FILE OPERATION FAILED", error=details)
                        logger.exception("Reset file operation failed for %s", source_path)

            _notify_progress(progress_callback, "Reset complete", 1.0)
            debug(
                "COMPLETE", action=action, scanned=len(ordered_paths),
                deleted=deleted_count, moved=moved_count, errors=len(file_errors),
            )
            if file_errors:
                raise RuntimeError(
                    "The database was reset, but some source files could not be "
                    "processed:\n" + "\n".join(file_errors[:10])
                )

            return {
                "file_count": len(ordered_paths),
                "action": action,
                "deleted_count": deleted_count,
                "moved_count": moved_count,
            }

    # Document metadata and document-level operations
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
                self.search_folder.parent / "deleted",
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

    # Public search API
    def search_with_report(
        self,
        terms: list[str],
        mode: TermSearchMode = "regex",
        *,
        fuzzy_distance: int = FUZZY_DISTANCE,
        max_distance_chars: int = MAX_DISTANCE_CHARS,
        snippet_context_chars: int = SNIPPET_CONTEXT_CHARS,
        max_results: int | None = MAX_RESULTS,
        max_results_per_document: int | None = MAX_RESULTS_PER_DOCUMENT,
        case_sensitive: Sequence[bool] | None = None,
        term_modes: Sequence[TermSearchMode] | None = None,
        progress_callback: ProgressCallback | None = None,
        selected_doc_ids: set[int] | None = None,
        literal_text_search: bool = False,
    ) -> tuple[list[SearchHit], SearchReport]:
        with self._index_lock:
            if self.index is None:
                return [], SearchReport(mode=mode)

            raw_case_flags = normalize_case_flags(case_sensitive, len(terms))
            raw_term_modes = normalize_term_modes(term_modes, len(terms), mode)
            clean_terms: list[str] = []
            clean_case_flags: list[bool] = []
            clean_term_modes: list[TermSearchMode] = []
            for term, sensitive, term_mode in zip(terms, raw_case_flags, raw_term_modes):
                cleaned = term.strip()
                if not cleaned:
                    continue
                clean_terms.append(cleaned)
                clean_case_flags.append(bool(sensitive))
                clean_term_modes.append(term_mode)

            config = SearchConfig(
                fuzzy_distance=fuzzy_distance,
                max_distance_chars=max_distance_chars,
                snippet_context_chars=snippet_context_chars,
                max_results_per_document=max_results_per_document,
                case_sensitive=tuple(clean_case_flags),
                term_modes=tuple(clean_term_modes),
                literal_text_search=bool(literal_text_search),
            )
            return search_index(
                self.index,
                clean_terms,
                mode,
                config,
                max_results,
                progress_callback=progress_callback,
                selected_doc_ids=selected_doc_ids,
            )

    def search(
        self,
        terms: list[str],
        mode: TermSearchMode = "regex",
        *,
        fuzzy_distance: int = FUZZY_DISTANCE,
        max_distance_chars: int = MAX_DISTANCE_CHARS,
        snippet_context_chars: int = SNIPPET_CONTEXT_CHARS,
        max_results: int | None = MAX_RESULTS,
        max_results_per_document: int | None = MAX_RESULTS_PER_DOCUMENT,
        case_sensitive: Sequence[bool] | None = None,
        term_modes: Sequence[TermSearchMode] | None = None,
        progress_callback: ProgressCallback | None = None,
        selected_doc_ids: set[int] | None = None,
        literal_text_search: bool = False,
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
            term_modes=term_modes,
            progress_callback=progress_callback,
            selected_doc_ids=selected_doc_ids,
            literal_text_search=literal_text_search,
        )
        return hits

# =============================================================================
# Result and timing report formatting
# =============================================================================

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
    if report.mode == "literal":
        total = max(report.total_ms, 0.000001)

        def literal_stage(label: str, value_ms: float) -> str:
            percent = 100.0 * value_ms / total
            return f"  - {label}: {value_ms:.2f} ms ({percent:.1f}% of wall time)"

        batches_by_worker: dict[str, dict] = {}
        for entry in report.literal_worker_batches:
            worker_name = str(entry.get("thread_name", "unknown worker"))
            worker = batches_by_worker.setdefault(
                worker_name,
                {
                    "batches": 0,
                    "files_assigned": 0,
                    "files_loaded": 0,
                    "matching_documents": 0,
                    "raw_candidate_documents": 0,
                    "raw_candidate_occurrences": 0,
                    "raw_count_ms": 0.0,
                    "load_ms": 0.0,
                    "processing_ms": 0.0,
                    "match_ms": 0.0,
                    "proximity_ms": 0.0,
                    "result_build_ms": 0.0,
                    "byte_normalize_ms": 0.0,
                    "byte_match_ms": 0.0,
                    "decode_ms": 0.0,
                    "bytes_searched": 0,
                    "thread_cpu_ms": 0.0,
                    "batch_elapsed_ms": 0.0,
                },
            )
            for field_name in worker:
                if field_name == "batches":
                    worker[field_name] += 1
                else:
                    worker[field_name] += entry.get(field_name, 0)

        load_sum_ms = sum(item["load_ms"] for item in batches_by_worker.values())
        processing_sum_ms = sum(
            item["processing_ms"] for item in batches_by_worker.values()
        )
        elapsed_sum_ms = sum(
            item["batch_elapsed_ms"] for item in batches_by_worker.values()
        )

        lines = [
            "Time report (literal text search):",
            f"  - Total search wall-clock (start to end): {report.total_ms:.2f} ms",
            "",
            "Preparation:",
            literal_stage("Load selected document IDs", report.literal_document_list_ms),
            literal_stage(
                "Prepare UTF-8 byte queries (ASCII-only case folding)",
                report.literal_query_prepare_ms,
            ),
            "",
            "Parallel batch processing:",
            f"  - Worker threads: {report.literal_worker_count:,} "
            "(pool size limited by detected physical CPU cores; threads are not CPU-pinned)",
            f"  - Batch size: {report.literal_batch_size:,} complete file(s) per task",
            f"  - Batches: {report.literal_batch_count:,}",
            f"  - Documents selected: {report.candidate_docs_after_selection:,}",
            f"  - Documents loaded: {report.source_text_docs:,}",
            literal_stage("Parallel batch phase wall-clock", report.literal_parallel_wall_ms),
            "  - Process CPU time during parallel phase: "
            f"{report.literal_parallel_process_cpu_ms:.2f} ms",
            "  - Average CPU cores used during parallel phase: "
            f"{report.literal_parallel_process_cpu_ms / max(report.literal_parallel_wall_ms, 0.000001):.2f}",
            f"  - Worker thread CPU time, summed: {report.literal_thread_cpu_ms:.2f} ms",
            f"  - Worker load time, summed across threads: {load_sum_ms:.2f} ms",
            f"  - Worker processing time, summed across threads: {processing_sum_ms:.2f} ms",
            f"  - Worker batch elapsed time, summed across threads: {elapsed_sum_ms:.2f} ms",
            "  - Note: summed worker times overlap; they are not total elapsed time and can exceed wall-clock time.",
            "",
            "Per-worker-thread timings (one pool thread per available worker slot):",
        ]
        if batches_by_worker:
            for worker_name, item in sorted(batches_by_worker.items()):
                lines.append(
                    f"  - {worker_name}: {item['batches']:,} batch(es), "
                    f"{item['files_assigned']:,} assigned / {item['files_loaded']:,} loaded, "
                    f"{item['raw_candidate_documents']:,} raw candidate(s), "
                    f"{item['matching_documents']:,} after proximity"
                )
                lines.append(
                    f"      load {item['load_ms']:.2f} ms; "
                    f"processing {item['processing_ms']:.2f} ms; "
                    f"batch elapsed Σ {item['batch_elapsed_ms']:.2f} ms"
                )
                lines.append(
                    f"      literal matching {item['match_ms']:.2f} ms; "
                    f"byte lowercase {item['byte_normalize_ms']:.2f} ms; "
                    f"byte matching {item['byte_match_ms']:.2f} ms; "
                    f"candidate UTF-8 decode {item['decode_ms']:.2f} ms"
                )
                lines.append(
                    f"      thread CPU {item['thread_cpu_ms']:.2f} ms; "
                    f"bytes searched {item['bytes_searched'] / (1024 * 1024):.2f} MiB; "
                    f"raw-count accounting {item['raw_count_ms']:.2f} ms; "
                    f"raw occurrences in candidates {item['raw_candidate_occurrences']:,}; "
                    f"proximity {item['proximity_ms']:.2f} ms; "
                    f"hit/result preparation {item['result_build_ms']:.2f} ms"
                )
        else:
            lines.append("  - No worker batches were executed.")

        lines.extend(
            [
                "",
                "Raw literal matches (before proximity filtering):",
                f"  - Raw occurrences across evaluated query terms: {report.literal_raw_occurrences_total:,}",
                f"  - Documents containing every query term: {report.literal_raw_candidate_docs:,}",
                f"  - Raw occurrences inside those candidate documents: {report.literal_raw_candidate_occurrences:,}",
                "  - Average raw occurrences per candidate document: "
                f"{report.literal_raw_candidate_occurrences / max(report.literal_raw_candidate_docs, 1):.2f}",
                "  - Maximum raw occurrences in one candidate document: "
                f"{report.literal_raw_candidate_max_occurrences:,}"
                + (
                    f" (doc_id={report.literal_raw_candidate_max_doc_id})"
                    if report.literal_raw_candidate_max_doc_id is not None else ""
                ),
                "  - Per-query occurrence counts:",
            ]
        )
        for term_stats in report.literal_raw_term_stats:
            query_label = repr(str(term_stats.get("query", "")))
            lines.append(
                f"      [{term_stats.get('query_index', 0)}] {query_label}: "
                f"{term_stats.get('occurrences', 0):,} occurrence(s) in "
                f"{term_stats.get('documents_with_matches', 0):,} document(s); "
                f"checked {term_stats.get('documents_checked', 0):,} document(s)"
            )
        if report.literal_raw_candidate_top_docs:
            lines.append(
                "  - Top 10 documents by raw occurrences "
                "(each listed document contains every query term):"
            )
            for item in report.literal_raw_candidate_top_docs:
                per_term = ", ".join(
                    f"{report.literal_raw_term_stats[index]['query']!r}={count:,}"
                    for index, count in enumerate(item["per_term"])
                )
                lines.append(
                    f"      doc_id={item['doc_id']}: {item['occurrences']:,} total "
                    f"raw occurrence(s) ({per_term}) — {item['path']}"
                )
        lines.extend(
            [
                "  - Raw-count accounting time (summed worker time): "
                f"{report.literal_raw_count_ms:.2f} ms",
                "  - Raw per-document summary aggregation (main thread): "
                f"{report.literal_raw_summary_ms:.2f} ms",
                "",
                "Proximity filtering:",
                f"  - Raw candidate documents checked: {report.proximity_docs_checked:,}",
                f"  - Documents passing proximity: {report.proximity_docs_with_hits:,}",
                f"  - Documents rejected by proximity: "
                f"{max(0, report.proximity_docs_checked - report.proximity_docs_with_hits):,}",
                "",
                "Finalization (main thread):",
                literal_stage("Sort matches and apply result limit", report.literal_sort_limit_ms),
                literal_stage("Reuse already-loaded result documents", report.documents_ms),
                literal_stage("Assemble final result objects", report.literal_result_assembly_ms),
                f"  - Documents passing proximity before global result limit: {report.literal_docs_before_limit:,}",
                f"  - Documents retained after global result limit: {report.literal_docs_after_limit:,}",
                f"  - Final documents returned: {report.final_documents:,}",
                f"  - Source files read: {report.runtime_source_file_reads:,}",
                f"  - Physical source bytes loaded: {report.runtime_source_bytes_read / (1024 * 1024):.2f} MiB",
                "  - SHA-256 calculations during literal search: 0 (skipped)",
                f"  - Documents byte-searched: {report.literal_byte_search_docs:,}",
                f"  - Raw candidate documents decoded for character positions/snippets: {report.literal_decoded_docs:,}",
                "  - Unicode regex/case-fold fallback: disabled (0)",
                "  - Logical bytes scanned across byte queries: "
                f"{report.literal_bytes_searched / (1024 * 1024):.2f} MiB",
                literal_stage(
                    "English/German lowercase byte-buffer preparation (summed worker time)",
                    report.literal_byte_normalize_ms,
                ),
                literal_stage(
                    "UTF-8 byte matching (summed worker time)",
                    report.literal_byte_match_ms,
                ),
                literal_stage(
                    "Candidate UTF-8 decoding and byte-to-character offset conversion (summed worker time)",
                    report.literal_decode_ms,
                ),
                "  - Case-sensitive matching: exact UTF-8 bytes via overlapping bytes.find()",
                "  - Case-insensitive matching: ASCII A-Z and German Ä/ä, Ö/ö, Ü/ü, ẞ/ß folded in UTF-8 bytes + overlapping bytes.find()",
                "  - ß and ss remain distinct; no general Unicode folding, transliteration, or normalization is performed",
                f"  - LanceDB document fallbacks: {report.runtime_lancedb_document_fallbacks:,}",
                "",
                "Timing interpretation:",
                "  - Total search and parallel batch phase are measured by start/end wall-clock timers.",
                "  - Per-worker values are summed task durations; concurrent work overlaps.",
            ]
        )
        if report.diagnostics:
            lines.extend(["", "Diagnostics:"])
            lines.extend(f"  - {diagnostic}" for diagnostic in report.diagnostics)
            lines.append(f"  - Full diagnostic log: {Path(SEARCH_DEBUG_LOG).absolute()}")
        return lines

    total = max(report.total_ms, 0.000001)

    def stage(label: str, value_ms: float) -> str:
        pct = 100.0 * value_ms / total
        return f"  - {label}: {value_ms:.2f} ms ({pct:.1f}%)"

    lines = [
        "Time report:",
        stage("Total search", report.total_ms),
        stage("Resolve variants", report.variant_resolution_ms),
    ]
    if report.mode in {"fuzzy", "both", "mixed"}:
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
    if report.wildcard_candidate_terms_examined:
        lines.append(
            "    - embedded wildcard vocabulary filtering: "
            f"{report.wildcard_candidate_lookup_ms:.2f} ms "
            f"({report.wildcard_candidate_terms_examined:,} candidate term(s) examined, "
            f"{report.wildcard_candidate_terms_matched:,} variant(s) retained)"
        )

    lines.append(stage("Tantivy candidate lookup", report.tantivy_ms))
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
        f"({report.candidate_docs_after_fts:,} document(s) left)"
    )

    if report.selection_applied:
        lines.append(
            stage("Selected-document filter", report.selection_filter_ms)
            + f" ({report.candidate_docs_before_selection:,} candidate(s) -> "
            f"{report.candidate_docs_after_selection:,}; "
            f"{report.selected_doc_id_count:,} selected unique-content document(s))"
        )

    lines.append(
        stage("Term-frequency lookup", report.frequency_ms)
        + f" ({report.frequency_rows:,} frequency row(s), "
        f"{report.frequency_docs:,} document(s))"
    )
    lines.append(
        f"    - all-query-term documents: {report.frequency_candidate_docs:,}"
    )
    if report.mode in {"fuzzy", "both", "mixed"}:
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
    lines.append(
        stage("Rarest-anchor position gate", report.anchor_gate_ms)
        + f" ({report.anchor_docs_before_gate:,} document(s) -> "
        + f"{report.anchor_docs_after_gate:,}; "
        + f"{report.anchor_component_builds:,} anchor component(s))"
    )
    lines.append(stage("Build positional matches", report.match_build_ms))
    if report.phrase_docs_before_prefilter:
        lines.append(
            f"    - phrase positional prefilter: {report.phrase_prefilter_ms:.2f} ms "
            f"({report.phrase_docs_before_prefilter:,} phrase-candidate document(s) -> "
            f"{report.phrase_docs_after_prefilter:,} geometrically possible document(s))"
        )
    if report.source_text_docs or report.source_text_ms:
        lines.append(
            stage("Fetch source text for matching", report.source_text_ms)
            + f" ({report.source_text_docs:,} document(s))"
        )
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
        f"  - Positional posting rows loaded: {report.runtime_posting_hits:,} across {report.runtime_posting_lookups:,} required term/document pair(s)",
        f"  - Positional posting cache: {report.posting_cache_hits:,} hit(s), {report.posting_cache_misses:,} miss(es), {report.posting_cache_entries:,} cached row(s), {report.posting_cache_bytes:,} bytes",
        f"  - Frequency rows fetched: {report.runtime_frequency_hits:,} across {report.runtime_frequency_lookups:,} term/document probe(s)",
        f"  - Fuzzy candidate terms enumerated in memory: {report.runtime_candidate_terms:,}",
        f"  - Embedded wildcard vocabulary candidates examined: {report.wildcard_candidate_terms_examined:,}",
        f"  - Result source-file reads: {report.runtime_source_file_reads:,}",
        f"  - LanceDB result-document fallbacks: {report.runtime_lancedb_document_fallbacks:,}",
        "",
        "LanceDB/PyArrow query instrumentation (nested inside the stages above):",
        f"  - DB predicate construction (where_in): {report.db_predicate_construction_ms:.2f} ms",
        f"  - DB metadata calls (e.g. count_rows): {report.db_metadata_calls:,} call(s), {report.db_metadata_ms:.2f} ms",
        f"  - table.search()/query setup: {report.db_query_setup_ms:.2f} ms",
        f"  - execution / to_arrow(): {report.db_execution_ms:.2f} ms",
        f"  - Arrow bytes returned: {report.db_bytes_returned:,} bytes",
        f"  - to_pylist() conversion: {report.db_to_pylist_ms:.2f} ms",
        f"  - nested Arrow-list to_pylist() conversion: {report.db_nested_to_pylist_ms:.2f} ms",
        f"  - Python object creation (inclusive in to_pylist()): {report.db_python_object_creation_ms:.2f} ms",
        f"  - Number of Arrow-producing DB queries issued: {report.db_queries:,}",
        f"  - Rows returned: {report.db_rows_returned:,}",
        f"  - Bytes returned: {report.db_bytes_returned:,}",
        f"  - Cache probe: {report.db_cache_cold_queries:,} cold / {report.db_cache_warm_queries:,} warm query(es) [process-lifecycle proxy]",
    ])
    if report.db_metadata_breakdown:
        lines.append("  - Per-query metadata breakdown:")
        for entry in report.db_metadata_breakdown:
            lines.append(
                f"    - {entry['operation']}: {entry['time_ms']:.2f} ms"
            )
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


def _format_repeated_values(counts: Mapping[str, int]) -> str:
    """Format distinct labels once, adding a multiplier for repetitions."""
    return "; ".join(
        f"{label} × {format_occurrence_count(count)}"
        if count > 1
        else label
        for label, count in counts.items()
    )


def _iter_match_annotations(occurrence: Mapping) -> Sequence[Mapping]:
    """Return component-level annotations, or the occurrence itself."""
    annotations = occurrence.get("component_annotations") or []
    return annotations if annotations else [occurrence]


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
            "Keyword modes: " + ", ".join(
                f"{term_matches[0]['query_term']!r}={term_mode}"
                for term_matches, term_mode in zip(hit.matches_by_term, hit.term_modes)
                if term_matches
            ),
            f"Proximity matches: {len(hit.chains)}",
            "Individual term occurrences:",
        ]

        document_wrapper_counts: dict[str, int] = {}
        for term_matches in hit.matches_by_term:
            query_term = term_matches[0]["query_term"] if term_matches else ""
            lines.append(
                f"  - '{query_term}': {format_occurrence_count(len(term_matches))} occurrence(s)"
            )
            for occurrence in term_matches:
                for annotation in _iter_match_annotations(occurrence):
                    wrapper = str(annotation.get("wrapper_flag", ""))
                    label = "ordinary/unwrapped" if not wrapper else wrapper
                    document_wrapper_counts[label] = (
                        document_wrapper_counts.get(label, 0) + 1
                    )

        if document_wrapper_counts:
            if set(document_wrapper_counts) == {"ordinary/unwrapped"}:
                only_count = next(iter(document_wrapper_counts.values()))
                lines.append(
                    "Wrapper summary: ordinary/unwrapped only "
                    f"({format_occurrence_count(only_count)} occurrence(s))"
                )
            else:
                lines.append(
                    "Wrapper summary: "
                    + _format_repeated_values(document_wrapper_counts)
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

            query_counts: dict[str, int] = {}
            matched_text_counts: dict[str, int] = {}
            special_wrapper_counts: dict[str, int] = {}
            for occurrence in ordered:
                query_term = str(occurrence.get("query_term", ""))
                query_counts[query_term] = query_counts.get(query_term, 0) + 1
                surface_text = content[occurrence["start"] : occurrence["end"]]
                matched_text_counts[surface_text] = (
                    matched_text_counts.get(surface_text, 0) + 1
                )
                for annotation in _iter_match_annotations(occurrence):
                    wrapper = str(annotation.get("wrapper_flag", ""))
                    if wrapper:
                        special_wrapper_counts[wrapper] = (
                            special_wrapper_counts.get(wrapper, 0) + 1
                        )

            lines += [
                f"  Match #{match_no}",
                f"  Position: {start_position}-{end_position}",
                f"  Span: {ordered[-1]['end'] - ordered[0]['start']} characters",
                "  Consecutive distances: "
                + ", ".join(str(d) for d in distances)
                + " characters",
                "  Query terms: " + ", ".join(query_counts),
                "  Matched text: " + _format_repeated_values(matched_text_counts),
            ]
            if special_wrapper_counts:
                lines.append(
                    "  Special wrappers: "
                    + _format_repeated_values(special_wrapper_counts)
                )
            lines.append(
                "  Snippet: " + build_snippet(content, ordered, snippet_context_chars)
            )
            if hit.mode in {"fuzzy", "both", "mixed"}:
                lines += [
                    "  Indexed terms: " + ", ".join(m["indexed_term"] for m in ordered),
                    "  Edit distances: " + ", ".join(str(m["edit_distance"]) for m in ordered),
                    "  Similarity: " + ", ".join(
                        f"{m['score']:.1f}%" for m in ordered
                    ),
                ]
            lines.append("")

    return "\n".join(lines).rstrip()

# =============================================================================
# Public exports
# =============================================================================

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
    "PhraseComponent",
    "PhraseSpec",
    "parse_phrase_query",
    "build_ordered_phrase_matches",
    "SearchMode",
    "TermSearchMode",
    "SearchHit",
    "SearchReport",
    "format_occurrence_count",
    "TextSearchEngine",
    "count_words",
    "find_proximity_chains_numpy",
    "format_results",
    "format_time_report",
]
