from __future__ import annotations

"""wxPython GUI; heavy search-engine imports are deferred until after startup."""

import math
import re
import shutil
import threading
import time
from pathlib import Path

import wx
import wx.stc as stc

# Keep startup lightweight: do not import engine.py here. It imports LanceDB,
# PyArrow, NumPy and RapidFuzz and is intentionally loaded after the first
# wx frame has been painted.
DB_PATH = "text_search.lancedb"
SEARCH_FOLDER = "txt"
FUZZY_DISTANCE = 1
MAX_DISTANCE_CHARS = 200
SNIPPET_CONTEXT_CHARS = 200
MAX_RESULTS_DEFAULT = "50" # "all"



class SearchProgressRing(wx.Panel):
    """Fixed circular progress ring: grey at 0%, blue at 100%."""

    def __init__(self, parent: wx.Window, size: tuple[int, int] = (22, 22)) -> None:
        super().__init__(parent, size=size)
        self._fraction = 0.0
        self.SetMinSize(size)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)

    def SetProgress(self, fraction: float) -> None:  # noqa: N802
        self._fraction = max(0.0, min(1.0, float(fraction)))
        self.Refresh(False)

    def Reset(self) -> None:  # noqa: N802
        self.SetProgress(0.0)

    def _on_paint(self, _event: wx.PaintEvent) -> None:
        dc = wx.AutoBufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()

        width, height = self.GetClientSize()
        diameter = max(6, min(width, height) - 4)
        pen_width = max(2, int(round(diameter * 0.15)))

        # One fixed integer bounding box is used for both the grey track and
        # the blue arc. Nothing about the circle's position or size changes
        # during progress.
        left = (width - diameter) // 2
        top = (height - diameter) // 2

        track_colour = wx.Colour(150, 150, 150)
        blue_colour = wx.Colour(0, 120, 215)

        dc.SetPen(wx.Pen(track_colour, pen_width))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawEllipse(left, top, diameter, diameter)

        fraction = self._fraction
        if fraction <= 0.0:
            return

        if fraction >= 1.0:
            # Exact same geometry as the grey track. At 100% there is no grey.
            dc.SetPen(wx.Pen(blue_colour, pen_width))
            dc.DrawEllipse(left, top, diameter, diameter)
            return

        # Draw the partial arc from 12 o'clock clockwise. The bounding box is
        # exactly the same fixed box used above, so the ring cannot drift.
        gc = wx.GraphicsContext.Create(dc)
        if gc is None:
            return

        gc.SetPen(wx.Pen(blue_colour, pen_width))
        cx = left + diameter / 2.0
        cy = top + diameter / 2.0
        radius = diameter / 2.0
        start_angle = -math.pi / 2.0
        end_angle = start_angle + (2.0 * math.pi * fraction)

        path = gc.CreatePath()
        # wx.GraphicsPath's clockwise flag is kept explicit.
        path.AddArc(cx, cy, radius, start_angle, end_angle, True)
        gc.StrokePath(path)


class SearchFrame(wx.Frame):
    def __init__(self) -> None:
        super().__init__(None, title="Text Search", size=(1200, 850))
        # Keep the GUI construction completely independent from database startup.
        # TextSearchEngine() can open LanceDB and build the static runtime caches,
        # so that work is deliberately deferred to a background thread.
        self.engine = None
        self.pending_paths: set[Path] = set()
        self.busy = False
        self.database_loading = True
        self._engine_module = None
        self._index_reset_timer = None
        self._folder_timer = wx.Timer(self)
        self._search_progress_timer = wx.Timer(self)
        self._search_progress_mode = "regex"
        self._search_reset_timer = None
        self._folder_snapshot: dict[str, tuple[int, int]] = {}
        self._keyword_fields: list[wx.TextCtrl] = []

        self._build_ui()
        self._set_database_action_state(False)
        self.search_progress.Reset()
        self.search_progress_label.SetLabel("Database loading — keywords can be entered")
        self.SetStatusText("Loading database ...")
        self.Centre()

        self.Bind(wx.EVT_TIMER, self._on_folder_timer, self._folder_timer)
        self.Bind(wx.EVT_TIMER, self._on_search_progress_timer, self._search_progress_timer)
        self.Bind(wx.EVT_CHAR_HOOK, self._on_char_hook)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._folder_timer.Start(3000)

    def _set_database_action_state(self, ready: bool) -> None:
        """Enable database-dependent actions only after the engine is ready."""
        self.btn_index.Enable(bool(ready))
        self.btn_choose.Enable(bool(ready))
        self.btn_reset_database.Enable(bool(ready))
        self.btn_search.Enable(bool(ready and self.engine is not None and self.engine.has_index))

    def _start_database_load(self) -> None:
        """Load the persistent index/caches without blocking the wx UI thread."""
        if not self.database_loading:
            return

        def worker() -> None:
            try:
                # Lazy import: this is the expensive native/database stack.
                import engine as engine_module

                engine_instance = engine_module.TextSearchEngine(DB_PATH, SEARCH_FOLDER)
                wx.CallAfter(
                    self._on_database_loaded,
                    engine_instance,
                    engine_module,
                    None,
                )
            except Exception as exc:  # pragma: no cover
                wx.CallAfter(self._on_database_loaded, None, None, exc)

        threading.Thread(
            target=worker,
            name="database-loader",
            daemon=True,
        ).start()

    def _on_database_loaded(
        self,
        engine,
        engine_module,
        error: Exception | None,
    ) -> None:
        """Complete database startup on the wx main thread."""
        self.database_loading = False

        if error is not None or engine is None:
            self.engine = None
            self._set_database_action_state(False)
            self.search_progress.Reset()
            self.search_progress_label.SetLabel("Database load failed")
            self.SetStatusText("Database load failed.")
            wx.MessageBox(
                str(error) if error is not None else "Database initialization failed.",
                "Database error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        self.engine = engine
        self._engine_module = engine_module
        self._set_database_action_state(True)

        # Do not delay the now-responsive GUI while the file list is refreshed.
        # It is scheduled after database initialization so the user can immediately
        # see/interact with the search controls.
        wx.CallAfter(self._finish_database_ui_refresh)

    def _finish_database_ui_refresh(self) -> None:
        if self.engine is None:
            return

        self._refresh_file_list()
        self._folder_snapshot = self._folder_state()
        self.search_progress.Reset()
        self.search_progress_label.SetLabel("Database ready")
        self.SetStatusText(
            "Database loaded."
            if self.engine.has_index
            else "No indexed database yet."
        )
        self.btn_search.Enable(self.engine.has_index)

    def _get_engine_module(self):
        if self._engine_module is None:
            raise RuntimeError("Database engine is still loading.")
        return self._engine_module

    def _build_ui(self) -> None:
        panel = wx.Panel(self)
        root = wx.BoxSizer(wx.VERTICAL)

        info = wx.StaticBoxSizer(wx.VERTICAL, panel, "Index")
        info_parent = info.GetStaticBox()
        info_grid = wx.FlexGridSizer(2, 2, 6, 8)
        info_grid.AddGrowableCol(1, 1)

        info_grid.Add(wx.StaticText(info_parent, label="Database:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.db_value = wx.StaticText(info_parent, label=str(Path(DB_PATH).absolute()))
        info_grid.Add(self.db_value, 1, wx.EXPAND)

        info_grid.Add(wx.StaticText(info_parent, label="Search folder:"), 0, wx.ALIGN_CENTER_VERTICAL)
        self.folder_value = wx.StaticText(info_parent, label=str(Path(SEARCH_FOLDER).absolute()))
        info_grid.Add(self.folder_value, 1, wx.EXPAND)
        info.Add(info_grid, 0, wx.EXPAND | wx.ALL, 8)

        index_button_row = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_index = wx.Button(info_parent, label="Index new files")
        self.btn_index.Bind(wx.EVT_BUTTON, self._on_index_new)
        index_button_row.Add(self.btn_index, 0, wx.RIGHT, 8)

        self.btn_choose = wx.Button(info_parent, label="Choose text file")
        self.btn_choose.Bind(wx.EVT_BUTTON, self._on_choose_files)
        index_button_row.Add(self.btn_choose, 0, wx.RIGHT, 8)

        self.btn_reset_database = wx.Button(info_parent, label="Reset database")
        self.btn_reset_database.Bind(wx.EVT_BUTTON, self._on_reset_database)
        self.btn_reset_database.SetToolTip(
            "Delete/reset the index and optionally delete or move all text files."
        )
        index_button_row.Add(self.btn_reset_database, 0)
        info.Add(index_button_row, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        index_progress_row = wx.BoxSizer(wx.HORIZONTAL)
        index_progress_row.Add(
            wx.StaticText(info_parent, label="Index progress:"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            8,
        )
        self.index_progress = wx.Gauge(info_parent, range=100, style=wx.GA_HORIZONTAL)
        index_progress_row.Add(self.index_progress, 1, wx.EXPAND)
        self.index_progress_label = wx.StaticText(info_parent, label="Ready")
        self.index_progress_label.SetMinSize((280, -1))
        index_progress_row.Add(self.index_progress_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.LEFT, 8)
        info.Add(index_progress_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)
        root.Add(info, 0, wx.EXPAND | wx.ALL, 8)

        files_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Files")
        files_parent = files_box.GetStaticBox()
        self.file_list = wx.ListCtrl(
            files_parent,
            style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SUNKEN,
        )
        self.file_list.InsertColumn(0, "Status", width=110)
        self.file_list.InsertColumn(1, "File", width=260)
        self.file_list.InsertColumn(2, "Words", width=100)
        self.file_list.InsertColumn(3, "Path", width=520)
        self.file_list.Bind(wx.EVT_RIGHT_DOWN, self._on_file_right_down)
        files_box.Add(self.file_list, 1, wx.EXPAND | wx.ALL, 6)
        root.Add(files_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        search_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Search")
        search_parent = search_box.GetStaticBox()

        # Left: keywords. Right: search parameters.
        input_grid = wx.FlexGridSizer(3, 4, 7, 12)
        input_grid.AddGrowableCol(1, 1)
        input_grid.AddGrowableCol(3, 1)

        keyword_fields = []
        for label, attr in (
            ("Keyword 1", "keyword_1"),
            ("Keyword 2", "keyword_2"),
            ("Keyword 3", "keyword_3"),
        ):
            input_grid.Add(wx.StaticText(search_parent, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
            field = wx.TextCtrl(search_parent)
            setattr(self, attr, field)
            input_grid.Add(field, 1, wx.EXPAND)
            keyword_fields.append(field)

            if attr == "keyword_1":
                self.fuzzy_distance = self._add_integer_field(
                    search_parent, input_grid, "FUZZY_DISTANCE", FUZZY_DISTANCE
                )
            elif attr == "keyword_2":
                self.max_distance_chars = self._add_integer_field(
                    search_parent, input_grid, "MAX_DISTANCE_CHARS", MAX_DISTANCE_CHARS
                )
            else:
                self.snippet_context_chars = self._add_integer_field(
                    search_parent, input_grid, "SNIPPET_CONTEXT_CHARS", SNIPPET_CONTEXT_CHARS
                )

        # TAB navigation between the keyword fields is handled at the frame
        # keyboard-hook level so wx's normal focus traversal cannot insert the
        # fuzzy-distance or other controls into this cycle.
        self._keyword_fields = keyword_fields

        search_box.Add(input_grid, 0, wx.EXPAND | wx.ALL, 8)

        control_row = wx.BoxSizer(wx.HORIZONTAL)
        control_row.Add(
            wx.StaticText(search_parent, label="Search mode"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            8,
        )
        self.mode = wx.Choice(search_parent, choices=["Regex", "Fuzz", "Both"])
        self.mode.SetSelection(0)
        self.mode.SetMinSize((105, -1))
        control_row.Add(self.mode, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 18)

        control_row.Add(
            wx.StaticText(search_parent, label="Max results"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            8,
        )
        self.max_results = wx.TextCtrl(
            search_parent,
            value=MAX_RESULTS_DEFAULT,
            size=(70, -1),
        )
        self.max_results.SetToolTip("Enter a positive number, or -1/all for unlimited results.")
        control_row.Add(self.max_results, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 18)

        control_row.Add(
            wx.StaticText(search_parent, label="Search:"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            6,
        )
        self.search_progress = SearchProgressRing(search_parent, size=(22, 22))
        control_row.Add(
            self.search_progress,
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            6,
        )
        self.search_progress_label = wx.StaticText(search_parent, label="Ready")
        self.search_progress_label.SetMinSize((210, -1))
        control_row.Add(
            self.search_progress_label,
            1,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            12,
        )

        self.timelogs = wx.CheckBox(search_parent, label="Timelogs")
        self.timelogs.SetValue(False)
        self.timelogs.SetToolTip("Show the detailed search timing report in the results.")
        control_row.Add(self.timelogs, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 10)

        self.btn_search = wx.Button(search_parent, label="Search")
        self.btn_search.Bind(wx.EVT_BUTTON, self._on_search)
        control_row.Add(self.btn_search, 0, wx.ALIGN_CENTER_VERTICAL)

        search_box.Add(control_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        root.Add(search_box, 0, wx.EXPAND | wx.ALL, 8)

        stats_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Search statistics")
        stats_parent = stats_box.GetStaticBox()

        stats_grid = wx.FlexGridSizer(3, 4, 5, 14)
        stats_grid.AddGrowableCol(1, 1)
        stats_grid.AddGrowableCol(2, 1)
        stats_grid.AddGrowableCol(3, 1)

        for label in ("", "Matches", "Individual term occurrences", "Search time"):
            stats_grid.Add(
                wx.StaticText(stats_parent, label=label),
                0,
                wx.ALIGN_CENTER_VERTICAL,
            )

        self.regex_stats = self._create_stats_row(stats_parent, stats_grid, "Regex")
        self.fuzzy_stats = self._create_stats_row(stats_parent, stats_grid, "Fuzzy")
        stats_box.Add(stats_grid, 0, wx.EXPAND | wx.ALL, 8)
        root.Add(stats_box, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        result_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Results")
        result_parent = result_box.GetStaticBox()

        self.output = stc.StyledTextCtrl(
            result_parent,
            style=wx.BORDER_SUNKEN | wx.CLIP_CHILDREN,
        )
        self.output.SetReadOnly(True)
        self.output.SetLexer(stc.STC_LEX_NULL)
        self.output.SetTabWidth(4)
        self.output.SetIndent(4)
        self.output.SetUseTabs(False)
        self.output.SetWrapMode(stc.STC_WRAP_WORD)
        self.output.SetUseHorizontalScrollBar(False)
        self.output.SetMarginWidth(0, 0)
        self.output.SetMarginWidth(1, 0)
        self.output.SetMarginWidth(2, 0)
        # Match highlighting is visual styling only. Scintilla's normal text
        # buffer remains unchanged, so copy can explicitly place plain text on
        # the clipboard without any formatting.
        self._output_match_style = 1
        self.output.StyleSetBackground(self._output_match_style, wx.Colour(255, 235, 140))
        self.output.StyleSetForeground(self._output_match_style, wx.BLACK)
        self.output.Bind(wx.EVT_KEY_DOWN, self._on_output_key_down)
        result_box.Add(self.output, 1, wx.EXPAND | wx.ALL, 6)
        root.Add(result_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 8)

        panel.SetSizer(root)
        self.CreateStatusBar()
        self._reset_stats()

    @staticmethod
    def _create_stats_row(
        parent: wx.Window,
        grid: wx.FlexGridSizer,
        mode_label: str,
    ) -> dict[str, wx.StaticText]:
        grid.Add(
            wx.StaticText(parent, label=mode_label),
            0,
            wx.ALIGN_CENTER_VERTICAL,
        )
        matches = wx.StaticText(parent, label="0")
        occurrences = wx.StaticText(parent, label="0")
        search_time = wx.StaticText(parent, label="0 ms")
        for control in (matches, occurrences, search_time):
            grid.Add(control, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL)
        return {
            "matches": matches,
            "occurrences": occurrences,
            "time": search_time,
        }

    @staticmethod
    def _stats_from_hits(
        hits: list[SearchHit],
        report: SearchReport,
    ) -> dict[str, object]:
        matches = sum(len(hit.chains) for hit in hits)

        term_occurrences: dict[str, int] = {}
        term_order: list[str] = []
        seen_terms: set[str] = set()

        for hit in hits:
            for term_matches in hit.matches_by_term:
                if not term_matches:
                    continue
                query_term = str(term_matches[0]["query_term"])
                normalized = query_term.casefold()
                if normalized not in seen_terms:
                    seen_terms.add(normalized)
                    term_order.append(query_term)
                term_occurrences[normalized] = (
                    term_occurrences.get(normalized, 0) + len(term_matches)
                )

        return {
            "matches": matches,
            "term_occurrences": term_occurrences,
            "term_order": term_order,
            "time_ms": report.total_ms,
        }

    @staticmethod
    def _format_occurrence_count(value: int) -> str:
        value = int(value)
        if value < 1000:
            return str(value)
        mantissa, exponent = f"{float(value):.1e}".split("e")
        return f"{mantissa}e{int(exponent)}"

    @classmethod
    def _set_stats_row(
        cls,
        row: dict[str, wx.StaticText],
        stats: dict[str, object],
    ) -> None:
        row["matches"].SetLabel(f"{int(stats['matches']):,}")

        occurrences = stats.get("term_occurrences", {})
        term_order = stats.get("term_order", [])
        if isinstance(occurrences, dict) and isinstance(term_order, list):
            values = [
                f"{term}: {cls._format_occurrence_count(int(occurrences.get(term.casefold(), 0)))}"
                for term in term_order
            ]
            row["occurrences"].SetLabel(",  ".join(values) if values else "—")
        else:
            row["occurrences"].SetLabel("—")

        row["time"].SetLabel(f"{float(stats['time_ms']):.1f} ms")

    def _reset_stats(self) -> None:
        empty = {
            "matches": 0,
            "term_occurrences": {},
            "term_order": [],
            "time_ms": 0.0,
        }
        self._set_stats_row(self.regex_stats, empty)
        self._set_stats_row(self.fuzzy_stats, empty)

    def _set_output_text_with_matches(self, marked_text: str) -> None:
        """Display marked results without marker characters and style matches.

        Scintilla positions are UTF-8 byte offsets, while Python regex/string
        positions are character offsets. Convert the recorded character ranges
        to byte offsets before applying the visual style so highlights stay
        aligned with the actual displayed characters, including non-ASCII text.
        """
        # The engine uses private-use Unicode characters as GUI-only match
        # delimiters. They are removed before text reaches the control.
        open_marker = "\ue000"
        close_marker = "\ue001"

        clean_chars: list[str] = []
        ranges: list[tuple[int, int]] = []
        match_start: int | None = None

        for char in marked_text:
            if char == open_marker:
                if match_start is None:
                    match_start = len(clean_chars)
                continue
            if char == close_marker:
                if match_start is not None:
                    end = len(clean_chars)
                    if end > match_start:
                        ranges.append((match_start, end))
                    match_start = None
                continue
            clean_chars.append(char)

        # Fail safe: never expose an unmatched internal marker.
        if match_start is not None:
            end = len(clean_chars)
            if end > match_start:
                ranges.append((match_start, end))

        text = "".join(clean_chars)
        self.output.SetReadOnly(False)
        self.output.SetText(text)
        self.output.EmptyUndoBuffer()

        self.output.StartStyling(0)
        self.output.SetStyling(self.output.GetTextLength(), 0)
        for char_start, char_end in ranges:
            byte_start = len(text[:char_start].encode("utf-8"))
            byte_end = len(text[:char_end].encode("utf-8"))
            if byte_end <= byte_start:
                continue
            self.output.StartStyling(byte_start)
            self.output.SetStyling(byte_end - byte_start, self._output_match_style)

        self.output.SetReadOnly(True)

    def _copy_output_plain_text(self) -> None:
        """Copy selected result text as plain text, never as styled formatting."""
        start = self.output.GetSelectionStart()
        end = self.output.GetSelectionEnd()
        if start == end:
            return

        selected = self.output.GetTextRange(start, end)
        data = wx.TextDataObject()
        data.SetText(selected)
        if wx.TheClipboard.Open():
            try:
                wx.TheClipboard.SetData(data)
                wx.TheClipboard.Flush()
            finally:
                wx.TheClipboard.Close()

    def _on_output_key_down(self, event: wx.KeyEvent) -> None:
        if event.ControlDown() and event.GetKeyCode() in (ord("C"), ord("A")):
            if event.GetKeyCode() == ord("A"):
                self.output.SetSelection(0, self.output.GetTextLength())
            else:
                self._copy_output_plain_text()
            return
        event.Skip()

    @staticmethod
    def _add_integer_field(
        parent: wx.Window,
        grid: wx.FlexGridSizer,
        label: str,
        value: int,
    ) -> wx.TextCtrl:
        grid.Add(wx.StaticText(parent, label=label), 0, wx.ALIGN_CENTER_VERTICAL)
        field = wx.TextCtrl(parent, value=str(value))
        grid.Add(field, 1, wx.EXPAND)
        return field

    def _read_non_negative_int(self, field: wx.TextCtrl, name: str) -> int:
        try:
            value = int(field.GetValue().strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    @staticmethod
    def _read_max_results(field: wx.TextCtrl) -> int | None:
        value = field.GetValue().strip().casefold()
        if value in {"all", "-1"}:
            return None
        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError("MAX_RESULTS must be a positive integer, -1, or all.") from exc
        if number < 1:
            raise ValueError("MAX_RESULTS must be a positive integer, -1, or all.")
        return number

    # ------------------------------------------------------------------
    # File list
    # ------------------------------------------------------------------

    def _refresh_file_list(self) -> None:
        if self.engine is None:
            self.file_list.DeleteAllItems()
            idx = self.file_list.InsertItem(0, "LOADING")
            self.file_list.SetItem(idx, 1, "Database")
            self.file_list.SetItem(idx, 2, "—")
            self.file_list.SetItem(idx, 3, "Loading index ...")
            self.file_list.SetItemData(idx, -1)
            return

        indexed = {
            str(Path(row["path"]).absolute()): row
            for row in self.engine.list_documents()
        }
        folder_files = {
            str(path.absolute()): path
            for path in Path(SEARCH_FOLDER).absolute().glob("*.txt")
        }
        pending = {str(path.absolute()): path for path in self.pending_paths}

        paths = sorted(
            set(indexed) | set(folder_files) | set(pending),
            key=lambda path: str(path).casefold(),
        )

        self.file_list.DeleteAllItems()
        for path_str in paths:
            path = Path(path_str)
            if path_str in indexed:
                status = "INDEXED" if path.exists() else "MISSING"
                word_count = indexed[path_str].get("word_count")
                doc_id = int(indexed[path_str]["id"])
            else:
                status = "NOT INDEXED"
                doc_id = -1
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    word_count = self._get_engine_module().count_words(content)
                except OSError:
                    word_count = None

            idx = self.file_list.InsertItem(self.file_list.GetItemCount(), status)
            self.file_list.SetItem(idx, 1, path.name)
            self.file_list.SetItem(idx, 2, f"{int(word_count):,}" if word_count is not None else "—")
            self.file_list.SetItem(idx, 3, path_str)
            self.file_list.SetItemData(idx, doc_id)

    def _folder_state(self) -> dict[str, tuple[int, int]]:
        folder = Path(SEARCH_FOLDER).absolute()
        state: dict[str, tuple[int, int]] = {}
        try:
            paths = folder.glob("*.txt")
        except OSError:
            return state
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            state[str(path.absolute())] = (int(stat.st_mtime_ns), int(stat.st_size))
        return state

    def _on_folder_timer(self, _event: wx.TimerEvent) -> None:
        if self.engine is None or self.busy:
            return

        current = self._folder_state()
        if current == self._folder_snapshot:
            return

        self._folder_snapshot = current
        self._refresh_file_list()

    def _on_close(self, _event: wx.CloseEvent) -> None:
        if self._folder_timer.IsRunning():
            self._folder_timer.Stop()
        if self._search_progress_timer.IsRunning():
            self._search_progress_timer.Stop()
        if self._index_reset_timer is not None:
            self._index_reset_timer.Stop()
            self._index_reset_timer = None
        if self._search_reset_timer is not None:
            self._search_reset_timer.Stop()
            self._search_reset_timer = None
        self.Destroy()

    def _on_file_right_down(self, event: wx.MouseEvent) -> None:
        item_index, _flags = self.file_list.HitTest(event.GetPosition())
        if item_index == wx.NOT_FOUND:
            return

        self.file_list.Select(item_index)
        self.file_list.Focus(item_index)
        doc_id = int(self.file_list.GetItemData(item_index))
        if doc_id < 0:
            return

        file_name = self.file_list.GetItemText(item_index, 1)
        path = self.file_list.GetItem(item_index, 3).GetText()

        menu = wx.Menu()
        delete_item = menu.Append(wx.ID_ANY, "Delete from database and move to 'deleted'")

        def on_delete(_event: wx.CommandEvent) -> None:
            self._on_delete_document(doc_id, file_name, path)

        menu.Bind(wx.EVT_MENU, on_delete, delete_item)
        try:
            self.PopupMenu(menu)
        finally:
            menu.Destroy()

    def _on_delete_document(self, doc_id: int, file_name: str, path: str) -> None:
        if self.engine is None:
            return

        confirmation = wx.MessageBox(
            f"Remove '{file_name}' from the database and move the original file to:\n\n"
            f"{Path(SEARCH_FOLDER).absolute() / 'deleted'}\n\n"
            f"Source: {path}",
            "Delete indexed file",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        )
        if confirmation != wx.YES:
            return

        if self._index_reset_timer is not None:
            self._index_reset_timer.Stop()
            self._index_reset_timer = None
        self._update_index_progress(0.0, "Deleting")

        def work():
            assert self.engine is not None
            return self.engine.delete_document(
                doc_id,
                progress_callback=self._queue_index_progress,
            )

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                self._update_index_progress(0.0, "Failed")
                self.SetStatusText("Deletion failed.")
                wx.MessageBox(str(error), "Deletion error", wx.OK | wx.ICON_ERROR)
                return

            self._refresh_file_list()
            self._folder_snapshot = self._folder_state()
            self._update_index_progress(1.0, "Ready")
            if self._index_reset_timer is not None:
                self._index_reset_timer.Stop()
            self._index_reset_timer = wx.CallLater(3000, self._reset_index_progress)

            deleted_path = result.get("deleted_path")
            if deleted_path:
                self.SetStatusText(f"Deleted '{file_name}' → {deleted_path}")
            else:
                self.SetStatusText(
                    f"Removed '{file_name}' from the database; source file was missing."
                )

        self._run_background(work, done)

    def _on_char_hook(self, event: wx.KeyEvent) -> None:
        """Cycle TAB strictly through Keyword 1, Keyword 2 and Keyword 3."""
        if event.GetKeyCode() == wx.WXK_TAB and self._keyword_fields:
            focused = wx.Window.FindFocus()
            try:
                current = self._keyword_fields.index(focused)
            except ValueError:
                event.Skip()
                return

            step = -1 if event.ShiftDown() else 1
            next_index = (current + step) % len(self._keyword_fields)
            target = self._keyword_fields[next_index]
            target.SetFocusFromKbd()
            target.SetInsertionPointEnd()
            return

        event.Skip()

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    def _update_search_progress(self, fraction: float, stage: str) -> None:
        # Use the engine's actual progress fraction. This is deliberately not a
        # time estimate: a large result set can take much longer to materialize.
        target = max(0.0, min(1.0, float(fraction)))
        current = self.search_progress._fraction
        if target < current:
            target = current
        self._search_progress_target = target
        self.search_progress_label.SetLabel(stage)

    def _start_search_indicator(self, mode: str, label: str = "Searching ...") -> None:
        if self._search_progress_timer.IsRunning():
            self._search_progress_timer.Stop()
        if self._search_reset_timer is not None:
            self._search_reset_timer.Stop()
            self._search_reset_timer = None
        self._search_progress_mode = mode
        self._search_progress_target = 0.0
        self.search_progress.Reset()
        self.search_progress_label.SetLabel(label)
        self._search_progress_timer.Start(25)

    def _on_search_progress_timer(self, _event: wx.TimerEvent) -> None:
        if not self._search_progress_timer.IsRunning():
            return
        current = self.search_progress._fraction
        target = getattr(self, "_search_progress_target", current)
        if target <= current:
            return
        # Smoothly approach the latest REAL engine progress without ever
        # overshooting it. The ring therefore cannot invent progress during a
        # long search or large result formatting pass.
        step = max(0.01, (target - current) * 0.35)
        self.search_progress.SetProgress(min(target, current + step))

    def _finish_search_indicator(self, label: str = "Ready", reset_delay_ms: int = 3000) -> None:
        if self._search_progress_timer.IsRunning():
            self._search_progress_timer.Stop()
        if self._search_reset_timer is not None:
            self._search_reset_timer.Stop()
            self._search_reset_timer = None
        self.search_progress.SetProgress(1.0)
        self.search_progress_label.SetLabel(label)
        if reset_delay_ms > 0:
            self._search_reset_timer = wx.CallLater(
                reset_delay_ms, self._reset_search_indicator
            )

    def _reset_search_indicator(self) -> None:
        self._search_reset_timer = None
        self.search_progress.Reset()
        self.search_progress_label.SetLabel("Ready")

    def _update_index_progress(self, fraction: float, stage: str) -> None:
        value = max(0, min(100, int(round(fraction * 100.0))))
        self.index_progress.SetValue(value)
        self.index_progress_label.SetLabel(f"{value}% — {stage}")

    def _queue_search_progress(self, stage: str, fraction: float) -> None:
        wx.CallAfter(self._update_search_progress, fraction, stage)

    def _queue_index_progress(self, stage: str, fraction: float) -> None:
        wx.CallAfter(self._update_index_progress, fraction, stage)

    @staticmethod
    def _scaled_progress(
        callback,
        offset: float,
        span: float,
    ):
        def wrapped(stage: str, fraction: float) -> None:
            callback(stage, offset + span * fraction)
        return wrapped

    # ------------------------------------------------------------------
    # Async helper
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool) -> None:
        self.busy = busy
        if busy:
            self.btn_index.Enable(False)
            self.btn_choose.Enable(False)
            self.btn_reset_database.Enable(False)
            self.btn_search.Enable(False)
            return
        self._set_database_action_state(
            self.engine is not None and not self.database_loading
        )

    def _run_background(self, work, on_done) -> None:
        if self.busy:
            return
        self._set_busy(True)
        self.SetStatusText("Working ...")

        def worker() -> None:
            try:
                result = work()
                wx.CallAfter(on_done, result, None)
            except Exception as exc:  # pragma: no cover
                wx.CallAfter(on_done, None, exc)

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Buttons
    # ------------------------------------------------------------------

    def _on_choose_files(self, _event: wx.CommandEvent) -> None:
        dialog = wx.FileDialog(
            self,
            "Choose text file(s)",
            wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE,
        )
        try:
            if dialog.ShowModal() != wx.ID_OK:
                return
            selected = [Path(path).absolute() for path in dialog.GetPaths()]
        finally:
            dialog.Destroy()

        folder = Path(SEARCH_FOLDER).absolute()

        def copy_files() -> list[Path]:
            folder.mkdir(parents=True, exist_ok=True)
            copied: list[Path] = []
            for source in selected:
                if source.suffix.casefold() != ".txt":
                    continue
                if self._inside(source, folder):
                    destination = source
                else:
                    destination = folder / source.name
                    # Keep an existing file with the same name intact.
                    if destination.exists() and destination.absolute() != source.absolute():
                        stem = destination.stem
                        suffix = destination.suffix
                        serial = 1
                        while True:
                            candidate = folder / f"{stem} ({serial}){suffix}"
                            if not candidate.exists():
                                destination = candidate
                                break
                            serial += 1
                    shutil.copy2(source, destination)
                copied.append(destination.absolute())
            return copied

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                self.SetStatusText("Copy failed.")
                wx.MessageBox(str(error), "Copy error", wx.OK | wx.ICON_ERROR)
                return
            self.pending_paths.update(result or [])
            self._folder_snapshot = self._folder_state()
            self._refresh_file_list()
            copied_count = len(result or [])
            self.SetStatusText(
                f"Copied {copied_count} text file(s) to {folder}."
                if copied_count
                else "No .txt files were selected."
            )

        self._run_background(copy_files, done)

    def _reset_index_progress(self) -> None:
        self._index_reset_timer = None
        self.index_progress.SetValue(0)
        self.index_progress_label.SetLabel("Ready")

    def _on_reset_database(self, _event: wx.CommandEvent) -> None:
        if self.engine is None:
            return

        class ResetDialog(wx.Dialog):
            def __init__(self, parent: wx.Window) -> None:
                super().__init__(parent, title="Reset database", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
                panel = wx.Panel(self)
                root = wx.BoxSizer(wx.VERTICAL)

                warning = wx.StaticText(
                    panel,
                    label=(
                        "Choose how to reset the search database.\n\n"
                        "The first two options are destructive and apply to indexed text files "
                        "as well as .txt files currently in the search folder."
                    ),
                )
                warning.Wrap(600)
                root.Add(warning, 0, wx.EXPAND | wx.ALL, 12)

                buttons = wx.BoxSizer(wx.HORIZONTAL)
                delete_btn = wx.Button(panel, label="Delete all files and database")
                move_btn = wx.Button(panel, label="Reset database, move files to delete folder")
                cancel_btn = wx.Button(panel, label="Cancel")
                buttons.Add(delete_btn, 0, wx.RIGHT, 8)
                buttons.Add(move_btn, 0, wx.RIGHT, 8)
                buttons.Add(cancel_btn, 0)
                root.Add(buttons, 0, wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.BOTTOM, 12)

                delete_btn.Bind(wx.EVT_BUTTON, lambda _e: self.EndModal(wx.ID_YES))
                move_btn.Bind(wx.EVT_BUTTON, lambda _e: self.EndModal(wx.ID_NO))
                cancel_btn.Bind(wx.EVT_BUTTON, lambda _e: self.EndModal(wx.ID_CANCEL))
                self.SetEscapeId(wx.ID_CANCEL)
                panel.SetSizer(root)
                self.SetSize((650, 170))
                self.CentreOnParent()

        dialog = ResetDialog(self)
        try:
            choice = dialog.ShowModal()
        finally:
            dialog.Destroy()

        if choice == wx.ID_CANCEL:
            return

        if self._index_reset_timer is not None:
            self._index_reset_timer.Stop()
            self._index_reset_timer = None
        self.pending_paths.clear()
        self._update_index_progress(0.0, "Preparing reset")

        action = "delete" if choice == wx.ID_YES else "move"

        def work():
            assert self.engine is not None
            return self.engine.reset_database(
                action,
                progress_callback=self._queue_index_progress,
            )

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                self._update_index_progress(0.0, "Failed")
                self.SetStatusText("Database reset failed.")
                wx.MessageBox(str(error), "Reset database error", wx.OK | wx.ICON_ERROR)
                return

            self._refresh_file_list()
            self._folder_snapshot = self._folder_state()
            self._update_index_progress(1.0, "Ready")
            if self._index_reset_timer is not None:
                self._index_reset_timer.Stop()
            self._index_reset_timer = wx.CallLater(3000, self._reset_index_progress)

            action_label = "deleted" if action == "delete" else "moved to deleted"
            file_count = int(result.get("file_count", 0)) if isinstance(result, dict) else 0
            self.SetStatusText(
                f"Database reset complete: {file_count} text file(s) {action_label}."
            )

        self._run_background(work, done)

    def _on_index_new(self, _event: wx.CommandEvent) -> None:
        pending = sorted(
            self.pending_paths,
            key=lambda path: str(path).casefold(),
        )
        folder = Path(SEARCH_FOLDER).absolute()

        if self._index_reset_timer is not None:
            self._index_reset_timer.Stop()
            self._index_reset_timer = None
        self._update_index_progress(0.0, "Starting")

        def work():
            if self.engine is None:
                raise RuntimeError("Database is still loading.")

            external = [path for path in pending if not self._inside(path, folder)]
            if external:
                sync_callback = self._scaled_progress(self._queue_index_progress, 0.0, 0.80)
                manual_callback = self._scaled_progress(self._queue_index_progress, 0.80, 0.20)
            else:
                sync_callback = self._queue_index_progress
                manual_callback = self._queue_index_progress

            sync_result = self.engine.sync_search_folder(progress_callback=sync_callback)
            manual_result = self.engine.index_files(
                external,
                progress_callback=manual_callback,
            ) if external else {
                "added": 0,
                "changed": 0,
                "removed": 0,
                "unchanged": 0,
            }
            return sync_result, manual_result

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                self._update_index_progress(0.0, "Failed")
                self.SetStatusText("Indexing failed.")
                wx.MessageBox(str(error), "Indexing error", wx.OK | wx.ICON_ERROR)
                return

            self.pending_paths.clear()
            self._refresh_file_list()
            sync_result, manual_result = result
            added = sync_result["added"] + manual_result["added"]
            changed = sync_result["changed"] + manual_result["changed"]
            removed = sync_result["removed"] + manual_result["removed"]
            self._update_index_progress(1.0, "Ready")
            if self._index_reset_timer is not None:
                self._index_reset_timer.Stop()
            self._index_reset_timer = wx.CallLater(3000, self._reset_index_progress)
            self._folder_snapshot = self._folder_state()
            self.SetStatusText(
                f"Index updated: {added} added, {changed} changed, {removed} removed."
            )

        self._run_background(work, done)

    def _on_search(self, _event: wx.CommandEvent) -> None:
        terms = [
            self.keyword_1.GetValue(),
            self.keyword_2.GetValue(),
            self.keyword_3.GetValue(),
        ]
        terms = [term.strip() for term in terms if term.strip()]
        if not terms:
            wx.MessageBox(
                "Enter at least one keyword.",
                "Search",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        try:
            fuzzy_distance = self._read_non_negative_int(
                self.fuzzy_distance, "FUZZY_DISTANCE"
            )
            max_distance_chars = self._read_non_negative_int(
                self.max_distance_chars, "MAX_DISTANCE_CHARS"
            )
            snippet_context_chars = self._read_non_negative_int(
                self.snippet_context_chars, "SNIPPET_CONTEXT_CHARS"
            )
            max_results = self._read_max_results(self.max_results)
        except ValueError as exc:
            wx.MessageBox(str(exc), "Search settings", wx.OK | wx.ICON_WARNING)
            return

        selection = self.mode.GetSelection()
        mode = ("regex", "fuzzy", "both")[selection]
        show_time_report = self.timelogs.GetValue()

        self._start_search_indicator(mode, "Searching ...")

        def run_search(
            search_mode: str,
            progress_callback,
        ) -> tuple[list, object, dict[str, object]]:
            if self.engine is None:
                raise RuntimeError("Database is still loading.")
            hits, report = self.engine.search_with_report(
                terms,
                search_mode,
                fuzzy_distance=fuzzy_distance,
                max_distance_chars=max_distance_chars,
                snippet_context_chars=snippet_context_chars,
                max_results=max_results,
                progress_callback=progress_callback,
            )
            return hits, report, self._stats_from_hits(hits, report)

        def work():
            if mode != "both":
                hits, report, stats = run_search(
                    mode,
                    self._queue_search_progress,
                )
                text = self._get_engine_module().format_results(
                    hits,
                    max_distance_chars=max_distance_chars,
                    snippet_context_chars=snippet_context_chars,
                    report=report if show_time_report else None,
                )
                return {
                    "text": text,
                    "regex": stats if mode == "regex" else None,
                    "fuzzy": stats if mode == "fuzzy" else None,
                }

            regex_hits, regex_report, regex_stats = run_search(
                "regex",
                self._scaled_progress(self._queue_search_progress, 0.0, 0.50),
            )
            fuzzy_hits, fuzzy_report, fuzzy_stats = run_search(
                "fuzzy",
                self._scaled_progress(self._queue_search_progress, 0.50, 0.50),
            )
            return {
                "text": self._format_both(
                    regex_hits,
                    fuzzy_hits,
                    max_distance_chars,
                    snippet_context_chars,
                    fuzzy_distance,
                    regex_report if show_time_report else None,
                    fuzzy_report if show_time_report else None,
                ),
                "regex": regex_stats,
                "fuzzy": fuzzy_stats,
            }

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                if self._search_progress_timer.IsRunning():
                    self._search_progress_timer.Stop()
                if self._search_reset_timer is not None:
                    self._search_reset_timer.Stop()
                    self._search_reset_timer = None
                self._reset_search_indicator()
                self.SetStatusText("Search failed.")
                wx.MessageBox(str(error), "Search error", wx.OK | wx.ICON_ERROR)
                return

            if mode == "both":
                total_ms = 0.0
                if result["regex"] is not None:
                    total_ms += float(result["regex"]["time_ms"])
                if result["fuzzy"] is not None:
                    total_ms += float(result["fuzzy"]["time_ms"])
            else:
                active_stats = result[mode]
                total_ms = float(active_stats["time_ms"]) if active_stats is not None else 0.0
            self._reset_stats()
            if result["regex"] is not None:
                self._set_stats_row(self.regex_stats, result["regex"])
            if result["fuzzy"] is not None:
                self._set_stats_row(self.fuzzy_stats, result["fuzzy"])

            self._set_output_text_with_matches(
                result["text"] or "No proximity matches found."
            )
            self.output.SetCurrentPos(0)
            self.output.SetAnchor(0)
            self.output.ScrollToStart()
            self.output.SetFocus()
            self._finish_search_indicator("Search complete.", reset_delay_ms=3000)
            self.SetStatusText("Search complete.")

        self._run_background(work, done)

    @staticmethod
    def _format_both(
        regex_hits: list,
        fuzzy_hits: list,
        max_distance_chars: int,
        snippet_context_chars: int,
        fuzzy_distance: int,
        regex_report,
        fuzzy_report,
    ) -> str:
        from engine import format_results

        blocks = [
            "REGEX RESULTS",
            "=" * 72,
            format_results(
                regex_hits,
                max_distance_chars=max_distance_chars,
                snippet_context_chars=snippet_context_chars,
                report=regex_report,
            ),
            "",
            "",
            f"FUZZ RESULTS (edit distance <= {fuzzy_distance})",
            "=" * 72,
            format_results(
                fuzzy_hits,
                max_distance_chars=max_distance_chars,
                snippet_context_chars=snippet_context_chars,
                report=fuzzy_report,
            ),
        ]
        return "\n".join(blocks)

    @staticmethod
    def _inside(path: Path, folder: Path) -> bool:
        try:
            path.relative_to(folder)
            return True
        except ValueError:
            return False


class SearchApp(wx.App):
    def OnInit(self) -> bool:  # noqa: N802
        frame = SearchFrame()
        frame.Show()
        self.SetTopWindow(frame)
        frame.keyword_1.SetFocus()

        # Let wx paint the window first. Database opening + static cache loading
        # starts only after the first UI event cycle, so the controls are visible
        # and keyword fields are usable while the database initializes.
        wx.CallAfter(frame._start_database_load)
        return True


def main() -> None:
    app = SearchApp(False)
    app.MainLoop()


if __name__ == "__main__":
    main()
