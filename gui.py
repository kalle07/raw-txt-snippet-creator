from __future__ import annotations

"""wxPython GUI; heavy search-engine imports are deferred until after startup."""

import os
import sys
import math
import re
import shutil
import threading
import time
from pathlib import Path

import wx
import wx.dataview as dv
import wx.stc as stc

# Keep startup lightweight: do not import engine.py here. It imports LanceDB,
# PyArrow, NumPy and RapidFuzz and is intentionally loaded after the first
# wx frame has been painted.
DB_PATH = "text_search.lancedb"
SEARCH_FOLDER = "txt"
FUZZY_DISTANCE = 1
MAX_DISTANCE_CHARS = 200
SNIPPET_CONTEXT_CHARS = 200
MAX_RESULTS_DEFAULT = "50"  # maximum number of ranked documents
MAX_RESULTS_PER_DOCUMENT_DEFAULT = "20"  # maximum displayed proximity matches per document




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
        self.engine = None
        self.pending_paths: set[Path] = set()
        self.busy = False
        self.database_loading = True
        self._engine_module = None

        self._index_reset_timer = None

        self._tree_context_event_bound = False
        self._folder_timer = wx.Timer(self)
        self._search_progress_timer = wx.Timer(self)
        self._search_progress_mode = "regex"
        self._search_reset_timer = None
        # Snapshot of the files actually synchronized with the database.
        self._folder_snapshot: dict[str, tuple[int, int]] = {}
        # Separate snapshot used only to avoid refreshing the tree repeatedly
        # while the user has not yet indexed externally added/changed files.
        self._folder_view_snapshot: dict[str, tuple[int, int]] = {}
        self._keyword_fields: list[wx.TextCtrl] = []
        self._case_sensitive_fields: list[wx.CheckBox] = []
        self._document_match_counts: dict[str, int] = {}
        # Last completed search state consumed by the modeless Document Browser.
        self._last_search_hits = None
        self._document_browser = None

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
        self.btn_choose_folder.Enable(bool(ready))
        self.btn_reset_database.Enable(bool(ready))
        self.btn_search.Enable(bool(ready and self.engine is not None and self.engine.has_index))
        self.btn_document_browser.Enable(bool(ready and self._last_search_hits))

    def _start_database_load(self) -> None:
        """Load the persistent index/caches without blocking the wx UI thread."""
        if not self.database_loading:
            return

        def worker() -> None:
            try:
                # Lazy import: this is the expensive native/database stack.
                import engine as engine_module

                engine_instance = engine_module.TextSearchEngine(
                    DB_PATH,
                    SEARCH_FOLDER,
                    progress_callback=self._queue_index_progress,
                )
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

        self._refresh_tree()
        # The manual-index snapshot must come from the database, not from the
        # filesystem. Files added before startup may exist on disk without being
        # indexed yet and must therefore remain visible to the Index button.
        self._folder_snapshot = self._indexed_folder_state()
        self._folder_view_snapshot = self._folder_state()

        # Show completed index progress briefly, then reset it.
        self._update_index_progress(1.0, "Database ready")
        self._schedule_index_progress_reset(delay_ms=1000)

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

        top_row = wx.BoxSizer(wx.HORIZONTAL)
        top_row.AddStretchSpacer(1)
        self.btn_document_browser = wx.Button(panel, label="document browser")
        self.btn_document_browser.SetToolTip(
            "Open the ranked documents and navigate through the matches from the last search."
        )
        self.btn_document_browser.Bind(wx.EVT_BUTTON, self._on_document_browser)
        self.btn_document_browser.Enable(False)
        top_row.Add(self.btn_document_browser, 0)
        root.Add(top_row, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 8)

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

        self.btn_choose = wx.Button(info_parent, label="Choose text file(s)")
        self.btn_choose.Bind(wx.EVT_BUTTON, self._on_choose_files)
        index_button_row.Add(self.btn_choose, 0, wx.RIGHT, 8)

        self.btn_choose_folder = wx.Button(info_parent, label="Choose folder")
        self.btn_choose_folder.Bind(wx.EVT_BUTTON, self._on_choose_folder)
        self.btn_choose_folder.SetToolTip(
            "Pick a directory; all .txt files inside it (recursively) are copied into the search folder."
        )
        index_button_row.Add(self.btn_choose_folder, 0, wx.RIGHT, 8)

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

        self.tree = dv.TreeListCtrl(
            files_parent,
            style=dv.TL_DEFAULT_STYLE | dv.TL_CHECKBOX,
        )
        self.tree.AppendColumn("Name", width=500)
        self.tree.AppendColumn("Size", width=100)
        self.tree.AppendColumn("Word Count", width=100)
        self.tree.AppendColumn("Status", width=150)
        
      
        self.tree.AppendColumn("Results", width=50)

        self.tree.Bind(dv.EVT_TREELIST_ITEM_CHECKED, self._on_tree_checked)
 
        
        # Row activation (clicking the row area)
        activated_event = getattr(dv, "EVT_TREELIST_ITEM_ACTIVATED", None)
        if activated_event is not None:
            self.tree.Bind(activated_event, self._on_tree_item_activated)

        self._bind_tree_click_handlers()
  

        files_box.Add(self.tree, 1, wx.EXPAND | wx.ALL, 6)
        root.Add(files_box, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        search_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Search")
        search_parent = search_box.GetStaticBox()

        # Three keyword fields (470 px) with an independent case-sensitivity
        # checkbox directly beside each field. Numeric search settings use compact
        # 100 px fields in the same rows.
        input_grid = wx.FlexGridSizer(3, 6, 7, 12)

        keyword_fields: list[wx.TextCtrl] = []
        case_sensitive_fields: list[wx.CheckBox] = []

        parameter_specs = (
            ("FUZZY_DISTANCE", FUZZY_DISTANCE),
            ("MAX_DISTANCE_CHARS", MAX_DISTANCE_CHARS),
            ("SNIPPET_CONTEXT_CHARS", SNIPPET_CONTEXT_CHARS),
        )

        keyword_specs = (
            ("Keyword 1", "keyword_1"),
            ("Keyword 2", "keyword_2"),
            ("Keyword 3", "keyword_3"),
        )

        parameter_attr_map = {
            "FUZZY_DISTANCE": "fuzzy_distance",
            "MAX_DISTANCE_CHARS": "max_distance_chars",
            "SNIPPET_CONTEXT_CHARS": "snippet_context_chars",
        }

        for (label, attr), (parameter_key, parameter_value) in zip(keyword_specs, parameter_specs):
            input_grid.Add(
                wx.StaticText(search_parent, label=label),
                0,
                wx.ALIGN_CENTER_VERTICAL,
            )

            field = wx.TextCtrl(search_parent, size=(470, -1))
            field.SetMinSize((470, -1))
            setattr(self, attr, field)
            input_grid.Add(field, 0, wx.ALIGN_CENTER_VERTICAL)
            keyword_fields.append(field)

            case_box = wx.CheckBox(search_parent, label="case-sensitive")
            case_box.SetValue(False)
            case_box.SetToolTip("Distinguish uppercase and lowercase letters for this keyword.")
            input_grid.Add(case_box, 0, wx.ALIGN_CENTER_VERTICAL)
            case_sensitive_fields.append(case_box)

            spacer = wx.StaticText(search_parent, label="")
            spacer.SetMinSize((80, -1))
            input_grid.Add(spacer, 0, wx.ALIGN_CENTER_VERTICAL)

            parameter_label_widget = wx.StaticText(search_parent, label=parameter_key)
            input_grid.Add(parameter_label_widget, 0, wx.ALIGN_CENTER_VERTICAL)

            parameter_field = wx.TextCtrl(
                search_parent,
                value=str(parameter_value),
                size=(100, -1),
            )
            parameter_field.SetMinSize((100, -1))
            input_grid.Add(parameter_field, 0, wx.ALIGN_CENTER_VERTICAL)

            setattr(self, parameter_attr_map[parameter_key], parameter_field)


        #self._keyword_fields = keyword_fields
        self._case_sensitive_fields = case_sensitive_fields

        search_box.Add(input_grid, 0, wx.EXPAND | wx.ALL, 8)

        # TAB navigation between the keyword fields is handled at the frame
        # keyboard-hook level so wx's normal focus traversal cannot insert the
        # fuzzy-distance or other controls into this cycle.
        self._keyword_fields = keyword_fields

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
            wx.StaticText(search_parent, label="Max Documents"),
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
            wx.StaticText(search_parent, label="Max Results / Document"),
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            8,
        )
        self.max_results_per_document = wx.TextCtrl(
            search_parent,
            value=MAX_RESULTS_PER_DOCUMENT_DEFAULT,
            size=(70, -1),
        )
        self.max_results_per_document.SetToolTip(
            "Maximum proximity matches shown per document. Enter a positive number, or -1/all for unlimited."
        )
        control_row.Add(
            self.max_results_per_document,
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            18,
        )

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
        """Display marked results without marker characters and style matches efficiently."""
        open_marker = "\ue000"
        close_marker = "\ue001"
        
        # Step 1: Strip markers and collect match ranges in character coordinates
        pattern = re.compile(re.escape(open_marker) + r"(.*?)" + re.escape(close_marker))
        
        clean_parts = []
        char_ranges = []  # List of (start_char, end_char) in the clean text
        
        last_end = 0
        current_clean_len = 0
        for match in pattern.finditer(marked_text):
            if match.start() > last_end:
                segment = marked_text[last_end:match.start()]
                clean_parts.append(segment)
                current_clean_len += len(segment)
            
            matched_content = match.group(1)
            clean_parts.append(matched_content)
            char_ranges.append((current_clean_len, current_clean_len + len(matched_content)))
            current_clean_len += len(matched_content)
            
            last_end = match.end()
        
        if last_end < len(marked_text):
            clean_parts.append(marked_text[last_end:])
            
        text = "".join(clean_parts)
        
        # Convert character ranges to byte offsets in ONE pass
        # Build a prefix sum array of byte lengths for each character
        # This is O(N) where N is the length of the clean text
        
        # Always make the control writable and reset it before replacing text.
        # This is important for the no-results case, where there are no match
        # ranges to style and the method would otherwise return early.
        self.output.SetReadOnly(False)

        # Temporarily disable word wrap to speed up layout
        was_wrap = self.output.GetWrapMode()
        self.output.SetWrapMode(stc.STC_WRAP_NONE)

        self.output.SetText(text)
        self.output.EmptyUndoBuffer()

        if not char_ranges:
            # No highlighted matches: clear any previous styling and restore state.
            self.output.StartStyling(0)
            self.output.SetStyling(self.output.GetTextLength(), 0)
            self.output.SetWrapMode(was_wrap)
            self.output.SetReadOnly(True)
            return

        # Efficiently compute byte offset for each character boundary.
        # We only need byte offsets at the boundaries of our ranges.
        boundary_chars = sorted(set(
            [start for start, _ in char_ranges] +
            [end for _, end in char_ranges]
        ))

        target_boundaries = set(boundary_chars)
        char_to_byte = {}
        current_byte_pos = 0
        for i, ch in enumerate(text):
            if i in target_boundaries:
                char_to_byte[i] = current_byte_pos
            current_byte_pos += len(ch.encode("utf-8"))

        # If the very end is a boundary, ensure it is captured.
        if len(text) in target_boundaries and len(text) not in char_to_byte:
            char_to_byte[len(text)] = current_byte_pos

        # Apply styling using pre-computed byte offsets.
        self.output.StartStyling(0)
        self.output.SetStyling(self.output.GetTextLength(), 0)

        for char_start, char_end in char_ranges:
            byte_start = char_to_byte.get(char_start, 0)
            byte_end = char_to_byte.get(char_end, current_byte_pos)
            if byte_end <= byte_start:
                continue
            self.output.StartStyling(byte_start)
            self.output.SetStyling(byte_end - byte_start, self._output_match_style)

        # Restore wrap mode.
        self.output.SetWrapMode(was_wrap)
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

    def _read_non_negative_int(self, field: wx.TextCtrl, name: str) -> int:
        try:
            value = int(field.GetValue().strip())
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if value < 0:
            raise ValueError(f"{name} must be >= 0.")
        return value

    @staticmethod
    def _read_max_results_per_document(field: wx.TextCtrl) -> int | None:
        value = field.GetValue().strip().casefold()
        if value in {"all", "-1"}:
            return None
        try:
            number = int(value)
        except ValueError as exc:
            raise ValueError(
                "MAX_RESULTS_PER_DOCUMENT must be a positive integer, -1, or all."
            ) from exc
        if number < 1:
            raise ValueError(
                "MAX_RESULTS_PER_DOCUMENT must be a positive integer, -1, or all."
            )
        return number

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
    # File tree (checkbox TreeListCtrl)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        return f"{size_bytes / (1024 * 1024):.1f} MB"


    def _populate_tree(self) -> None:
        """Reconcile the tree with current SEARCH_FOLDER contents and pending files.

        Preserves checkbox state on unchanged items by updating in-place rather
        than rebuilding from scratch.
        """
        if self.engine is None:
            # During initial load, just show a placeholder
            self.tree.DeleteAllItems()
            root = self.tree.GetRootItem()
            self.tree.AppendItem(root, "Loading index …")
            return

        folder = Path(SEARCH_FOLDER).absolute()
        indexed: dict[str, dict] = {
            str(Path(row["path"]).absolute()): row
            for row in self.engine.list_documents()
        }
        pending: set[Path] = set(self.pending_paths)

        # Build the desired state: map of absolute_path -> (name, size_str, word_count_str, status_text)
        desired_files: dict[str, tuple[str, str, str, str]] = {}

        # External pending files (outside search folder)
        external_paths = sorted(
            p for p in pending if not self._inside(p, folder)
        )
        for path in external_paths:
            status_text = "Pending" if str(path.absolute()) not in indexed else "Indexed"
            try:
                size_str = self._format_size(int(path.stat().st_size))
            except OSError:
                size_str = "—"
            word_count_str = "—"
            if path.suffix.casefold() == ".txt":
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    wc = self._get_engine_module().count_words(content)
                    word_count_str = str(wc) if wc is not None else "—"
                except (OSError, RuntimeError):
                    word_count_str = "—"
            desired_files[str(path.absolute())] = (
                path.name, size_str, word_count_str, status_text
            )

        # Files inside SEARCH_FOLDER
        try:
            entries = sorted(folder.rglob("*.txt"), key=lambda p: str(p).casefold())
        except OSError:
            entries = []

        for entry in entries:
            if self._is_deleted_path(entry, folder):
                continue
            abs_path = str(entry.absolute())
            if abs_path in indexed:
                status_text = "Indexed" if entry.exists() else "MISSING"
                wc = indexed[abs_path].get("word_count")
                word_count_str = str(wc) if wc is not None else "—"
            elif entry in pending or abs_path in {str(p) for p in pending}:
                status_text = "Pending"
                try:
                    content = entry.read_text(encoding="utf-8", errors="ignore")
                    wc = self._get_engine_module().count_words(content)
                    word_count_str = str(wc) if wc is not None else "—"
                except (OSError, RuntimeError):
                    word_count_str = "—"
            else:
                status_text = "Not indexed"
                word_count_str = "—"

            try:
                size_str = self._format_size(int(entry.stat().st_size))
            except OSError:
                size_str = "—"

            desired_files[abs_path] = (entry.name, size_str, word_count_str, status_text)

        # Now reconcile the tree
        self._reconcile_tree(desired_files)


    def _reconcile_tree(self, desired_files: dict[str, tuple[str, str, str, str]]) -> None:
        """Update the tree to match desired_files, preserving checkbox state."""
        internal_root = self.tree.GetRootItem()

        existing_items: dict[str, dv.TreeListItem] = {}

        def collect_existing(node: dv.TreeListItem) -> None:
            path = self._get_item_path(node)
            if path is not None:
                existing_items[str(path.absolute())] = node

            child = self.tree.GetFirstChild(node)
            while child.IsOk():
                collect_existing(child)
                child = self.tree.GetNextSibling(child)

        child = self.tree.GetFirstChild(internal_root)
        while child.IsOk():
            collect_existing(child)
            child = self.tree.GetNextSibling(child)

        desired_paths = set(desired_files)
        existing_paths = set(existing_items)

        to_remove = existing_paths - desired_paths
        for path_str in to_remove:
            item = existing_items.get(path_str)
            if item is not None and item.IsOk():
                self.tree.DeleteItem(item)

        if to_remove or (desired_paths - existing_paths):
            checkbox_states: dict[str, bool] = {}
            for path_str, item in existing_items.items():
                if path_str in desired_paths:
                    checkbox_states[path_str] = (
                        self.tree.GetCheckedState(item) == wx.CHK_CHECKED
                    )

            self.tree.DeleteAllItems()
            self._build_tree_from_desired(desired_files, Path(SEARCH_FOLDER).absolute())

            def restore_checkboxes(node: dv.TreeListItem) -> None:
                path = self._get_item_path(node)
                if path is not None:
                    path_str = str(path.absolute())
                    if checkbox_states.get(path_str):
                        self.tree.CheckItem(node, True)

                child = self.tree.GetFirstChild(node)
                while child.IsOk():
                    restore_checkboxes(child)
                    child = self.tree.GetNextSibling(child)

            child = self.tree.GetFirstChild(internal_root)
            while child.IsOk():
                restore_checkboxes(child)
                child = self.tree.GetNextSibling(child)
        else:
            for path_str, item in existing_items.items():
                if path_str in desired_files:
                    _, size_str, word_count_str, status_text = desired_files[path_str]
                    self.tree.SetItemText(item, 1, size_str)
                    self.tree.SetItemText(item, 2, word_count_str)
                    self.tree.SetItemText(item, 3, status_text)

        self._update_tree_results()



    def _set_item_path(self, item: dv.TreeListItem, path: Path) -> None:
        """Attach the authoritative filesystem path to a tree item."""
        self.tree.SetItemData(item, path)

    def _get_item_path(self, item: dv.TreeListItem | None) -> Path | None:
        """Return the filesystem path attached to a tree item, if any."""
        if item is None or not item.IsOk():
            return None

        data = self.tree.GetItemData(item)
        if isinstance(data, Path):
            return data
        if isinstance(data, str):
            return Path(data)
        return None



    def _build_tree_from_desired(
        self,
        desired_files: dict[str, tuple[str, str, str, str]],
        folder: Path,
    ) -> None:
        """Build the tree structure from desired_files mapping."""
        internal_root = self.tree.GetRootItem()

        external_items: dict[str, tuple[str, str, str, str]] = {}
        internal_items: dict[str, tuple[str, str, str, str]] = {}

        for path_str, values in desired_files.items():
            path = Path(path_str)
            if not self._inside(path, folder):
                external_items[path_str] = values
            else:
                internal_items[path_str] = values

        # External files are root-level leaves.
        for path_str in sorted(external_items):
            name, size_str, wc_str, status = external_items[path_str]
            item = self.tree.AppendItem(internal_root, name)
            self.tree.SetItemText(item, 1, size_str)
            self.tree.SetItemText(item, 2, wc_str)
            self.tree.SetItemText(item, 3, status)
            self.tree.SetItemText(item, 4, "")
            self._set_item_path(item, Path(path_str))

        if not internal_items:
            return

        folders: dict[str, list[tuple[str, str, str, str]]] = {}
        for path_str, values in internal_items.items():
            path = Path(path_str)
            parent_dir = str(path.parent)
            folders.setdefault(parent_dir, []).append(
                (path.name, values[1], values[2], values[3])
            )

        def build_folder(dir_path: str, parent_item: dv.TreeListItem) -> dv.TreeListItem | None:
            dir_obj = Path(dir_path)
            dir_name = dir_obj.name or str(dir_obj)

            item = self.tree.AppendItem(parent_item, dir_name)
            self.tree.SetItemText(item, 1, self._format_size(int(dir_obj.stat().st_size)))
            self.tree.SetItemText(item, 2, "")
            self.tree.SetItemText(item, 3, "")
            self.tree.SetItemText(item, 4, "")
            self._set_item_path(item, dir_obj)

            for name, fsize, fwc, fstatus in folders.get(dir_path, []):
                file_path = dir_obj / name
                file_item = self.tree.AppendItem(item, name)
                self.tree.SetItemText(file_item, 1, fsize)
                self.tree.SetItemText(file_item, 2, fwc)
                self.tree.SetItemText(file_item, 3, fstatus)
                self.tree.SetItemText(file_item, 4, "")
                self._set_item_path(file_item, file_path)

            if dir_obj.is_dir():
                try:
                    sub_dirs = sorted(
                        (p for p in dir_obj.iterdir() if p.is_dir()),
                        key=lambda p: p.name.casefold(),
                    )
                except OSError:
                    sub_dirs = []

                for sub_dir in sub_dirs:
                    if sub_dir.name == "deleted":
                        continue
                    build_folder(str(sub_dir), item)

            return item

        main_folder_item = build_folder(str(folder), internal_root)
        if main_folder_item is not None and main_folder_item.IsOk():
            self.tree.Expand(main_folder_item)
        self.tree.Expand(internal_root)



    def _refresh_tree(self) -> None:
        wx.CallAfter(self._populate_tree)


    def _update_tree_results(self) -> None:
        """Update only the Results column for file items."""
        if self.tree is None:
            return

        internal_root = self.tree.GetRootItem()

        def update_item(item: dv.TreeListItem) -> None:
            path = self._get_item_path(item)
            if path is None:
                return

            if not path.is_file() or path.suffix.casefold() != ".txt":
                return

            doc_path_str = str(path.absolute())
            match_count = self._document_match_counts.get(doc_path_str, 0)
            results_str = str(match_count) if match_count > 0 else ""
            self.tree.SetItemText(item, 4, results_str)

        def walk(node: dv.TreeListItem) -> None:
            update_item(node)

            child = self.tree.GetFirstChild(node)
            while child.IsOk():
                walk(child)
                child = self.tree.GetNextSibling(child)

        child = self.tree.GetFirstChild(internal_root)
        while child.IsOk():
            walk(child)
            child = self.tree.GetNextSibling(child)



    def _bind_tree_click_handlers(self) -> None:
        """Bind mouse events to handle row clicks for checkbox toggling."""
        self.tree.Bind(wx.EVT_LEFT_DOWN, self._on_tree_left_down)

        context_menu_event = getattr(dv, "EVT_TREELIST_ITEM_CONTEXT_MENU", None)
        if context_menu_event is not None:
            self.tree.Bind(context_menu_event, self._on_tree_item_context_menu)
            self._tree_context_event_bound = True
        else:
            self._tree_context_event_bound = False
            self.tree.Bind(wx.EVT_RIGHT_DOWN, self._on_tree_right_down)


    def _is_tree_toggleable_row(self, item: dv.TreeListItem | None) -> bool:
        """Return True for rows whose checkbox may be toggled by row clicks."""
        if item is None or not item.IsOk():
            return False

        if self._is_tree_file_item(item):
            return True

        path = self._get_item_path(item)
        return path is not None and path.is_dir()

   

    def _on_tree_left_down(self, event: wx.MouseEvent) -> None:
        """Handle left mouse button down on tree items."""
        item = self._get_tree_item_from_event(event)

        if item is None or not item.IsOk():
            event.Skip()
            return

        if not self._is_tree_toggleable_row(item):
            event.Skip()
            return

        # Determine if the click was on the checkbox itself.
        source = event.GetEventObject()
        clicked_on_checkbox = False

        if isinstance(source, wx.Window):
            try:
                screen_pos = source.ClientToScreen(event.GetPosition())
                main_window = self.tree
                get_main = getattr(self.tree, "GetMainWindow", None)
                if callable(get_main):
                    try:
                        candidate = get_main()
                        if candidate is not None:
                            main_window = candidate
                    except Exception:
                        pass

                pos = main_window.ScreenToClient(screen_pos)

                # If the click is inside the first column, treat it as a checkbox click.
                col_width = self.tree.GetColumnWidth(0)
                clicked_on_checkbox = pos.x < col_width

            except Exception:
                clicked_on_checkbox = False

        if clicked_on_checkbox:
            # Let wx handle checkbox clicks natively.
            event.Skip()
            return

        # Click was on the row area, not the checkbox.
        try:
            current_state = int(self.tree.GetCheckedState(item))
        except Exception:
            event.Skip()
            return

        new_state = wx.CHK_UNCHECKED if current_state == wx.CHK_CHECKED else wx.CHK_CHECKED
        self._apply_tree_check_state(item, new_state)

        # Do not skip the event so wx can still select the row normally.




    def _is_tree_file_item(self, item: dv.TreeListItem | None) -> bool:
        """Return True for a leaf row representing a .txt file."""
        if item is None or not item.IsOk():
            return False
        name = self.tree.GetItemText(item, 0)
        if not name.casefold().endswith(".txt"):
            return False
        child = self.tree.GetFirstChild(item)
        return not child.IsOk()


    def _on_tree_item_activated(self, event: dv.TreeListEvent) -> None:
        """Handle row activation clicks on toggleable rows."""
        item = event.GetItem()
        if not item.IsOk():
            event.Skip()
            return

        if not self._is_tree_toggleable_row(item):
            event.Skip()
            return

        try:
            current_state = int(self.tree.GetCheckedState(item))
        except Exception:
            event.Skip()
            return

        new_state = (
            wx.CHK_UNCHECKED
            if current_state == wx.CHK_CHECKED
            else wx.CHK_CHECKED
        )
        self._apply_tree_check_state(item, new_state)
        event.Skip()



    def _apply_tree_check_state(
        self,
        item: dv.TreeListItem,
        state: int,
    ) -> None:
        """Apply a checkbox state to a file or folder row."""
        if not item.IsOk():
            return

        if self._is_tree_file_item(item):
            self.tree.CheckItem(item, state)
            return

        # Folder rows propagate to children and subfolders.
        self.tree.CheckItemRecursively(item, state)



    def _on_tree_checked(self, event: dv.TreeListEvent) -> None:
        """Handle native checkbox toggles."""
        item = event.GetItem()
        if not item.IsOk():
            event.Skip()
            return

        if self._is_tree_file_item(item):
            event.Skip()
            return

        # For folders, propagate the new state to children and subfolders.
        try:
            current_state = int(self.tree.GetCheckedState(item))
        except Exception:
            event.Skip()
            return

        self._apply_tree_check_state(item, current_state)
        event.Skip()





    def _get_selected_leaf_paths(self) -> set[Path]:
        """Return absolute paths of all checked leaf .txt files in the tree."""
        selected: set[Path] = set()
        internal_root = self.tree.GetRootItem()

        def walk(node: dv.TreeListItem) -> None:
            if self.tree.GetCheckedState(node) == wx.CHK_CHECKED:
                path = self._get_item_path(node)
                if (
                    path is not None
                    and path.is_file()
                    and path.suffix.casefold() == ".txt"
                ):
                    selected.add(path.absolute())

            child = self.tree.GetFirstChild(node)
            while child.IsOk():
                walk(child)
                child = self.tree.GetNextSibling(child)

        child = self.tree.GetFirstChild(internal_root)
        while child.IsOk():
            walk(child)
            child = self.tree.GetNextSibling(child)

        return selected


    def _folder_state(self) -> dict[str, tuple[int, int]]:
        """Return the current filesystem state of searchable .txt files."""
        folder = Path(SEARCH_FOLDER).absolute()
        state: dict[str, tuple[int, int]] = {}
        try:
            paths = list(folder.rglob("*.txt"))
        except OSError:
            return state
        for path in paths:
            if self._is_deleted_path(path, folder):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            state[str(path.absolute())] = (int(stat.st_mtime_ns), int(stat.st_size))
        return state

    def _indexed_folder_state(self) -> dict[str, tuple[int, int]]:
        """Return filesystem state for files that are actually in the database.

        This is deliberately different from ``_folder_state()``.  A file that
        exists on disk but is not indexed must NOT appear in the indexed
        snapshot; otherwise the Index New Files button can incorrectly decide
        that there is nothing to do.  Missing indexed files are retained with a
        sentinel state so their removal is also detected.
        """
        state: dict[str, tuple[int, int]] = {}
        if self.engine is None:
            return state

        folder = Path(SEARCH_FOLDER).absolute()
        for row in self.engine.list_documents():
            try:
                path = Path(str(row["path"])).absolute()
                # Use self._instead of module-level path_in_folder
                if not self._inside(path, folder) or self._is_deleted_path(path, folder):
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    # Keep the indexed path in the snapshot so the missing file
                    # is seen as a removal on the next manual synchronization.
                    state[str(path)] = (-1, -1)
                    continue
                state[str(path)] = (int(stat.st_mtime_ns), int(stat.st_size))
            except (KeyError, TypeError, ValueError):
                continue
        return state

    def _on_folder_timer(self, _event: wx.TimerEvent) -> None:
        if self.engine is None or self.busy:
            return

        current = self._folder_state()
        # The timer only tracks what is currently visible on disk.  Do NOT copy
        # this into _folder_snapshot: that snapshot represents the last state
        # actually synchronized with the database.  Otherwise a newly created
        # file gets noticed by the timer, the snapshot advances, and the manual
        # "Index new files" button incorrectly concludes that nothing changed.
        if current == self._folder_view_snapshot:
            return

        self._folder_view_snapshot = current
        self._refresh_tree()

    def _on_close(self, _event: wx.CloseEvent) -> None:
        if self._folder_timer.IsRunning():
            self._folder_timer.Stop()
        if self._search_progress_timer.IsRunning():
            self._search_progress_timer.Stop()
        if self._document_browser is not None:
            try:
                self._document_browser.Destroy()
            except RuntimeError:
                pass
            self._document_browser = None
        self._index_reset_timer = None
        self.Destroy()




    def _get_tree_item_from_event(self, event: wx.MouseEvent) -> dv.TreeListItem | None:
        """Return the tree item under the mouse event, if any."""
        source = event.GetEventObject()
        if not isinstance(source, wx.Window):
            return None

        try:
            screen_pos = source.ClientToScreen(event.GetPosition())
        except Exception:
            return None

        main_window = self.tree
        get_main_window = getattr(self.tree, "GetMainWindow", None)
        if callable(get_main_window):
            try:
                candidate = get_main_window()
                if candidate is not None:
                    main_window = candidate
            except Exception:
                pass

        try:
            pos = main_window.ScreenToClient(screen_pos)
        except Exception:
            return None

        try:
            hit = self.tree.HitTest(pos)
        except Exception:
            return None

        item = hit[0] if isinstance(hit, tuple) else hit
        if item is None or not item.IsOk():
            return None

        return item


    def _show_tree_item_menu(
        self,
        item: dv.TreeListItem,
        popup_window: wx.Window | None = None,
        popup_pos: tuple[int, int] | None = None,
    ) -> None:
        """Show a context menu for either a text file or a folder tree."""
        if item is None or not item.IsOk():
            return

        abs_path = self._get_item_path(item)
        if abs_path is None:
            return

        is_file = abs_path.is_file() and abs_path.suffix.casefold() == ".txt"
        is_folder = abs_path.is_dir()

        search_folder = Path(SEARCH_FOLDER).absolute()
        if is_folder:
            if abs_path == search_folder or not self._inside(abs_path, search_folder):
                return
            if self._is_deleted_path(abs_path, search_folder):
                return
        elif not is_file:
            return

        self.tree.Select(item)

        menu = wx.Menu()

        if is_file:
            remove_id = menu.Append(
                wx.ID_ANY,
                "Remove from database and move to 'deleted'",
            )
            menu.Bind(
                wx.EVT_MENU,
                lambda _e, path=abs_path: self._on_remove_from_db_and_move(path),
                remove_id,
            )

            menu.AppendSeparator()

            open_id = menu.Append(wx.ID_ANY, "Open with default viewer")
            menu.Bind(
                wx.EVT_MENU,
                lambda _e, path=abs_path: self._on_open_file(path),
                open_id,
            )

        elif is_folder:
            remove_tree_id = menu.Append(
                wx.ID_ANY,
                "Remove folder tree from database and move text files to 'deleted'",
            )
            menu.Bind(
                wx.EVT_MENU,
                lambda _e, folder=abs_path: self._on_remove_folder_tree_and_move(folder),
                remove_tree_id,
            )

        if popup_window is None:
            popup_window = self.tree
        if popup_pos is None:
            screen_pos = wx.GetMousePosition()
            popup_pos = popup_window.ScreenToClient(screen_pos)

        try:
            popup_window.PopupMenu(menu, popup_pos)
        finally:
            menu.Destroy()



    def _on_tree_item_context_menu(self, event: dv.TreeListEvent) -> None:
        """Preferred TreeListCtrl context-menu event."""
        item = event.GetItem()
        if not item.IsOk():
            event.Skip()
            return

        self._show_tree_item_menu(item)



    def _on_tree_right_down(self, event: wx.MouseEvent) -> None:
        """Fallback for ports that deliver right-clicks as ordinary mouse events."""
        if self._tree_context_event_bound:
            event.Skip()
            return

        item = self._get_tree_item_from_event(event)
        self._show_tree_item_menu(item)






    def _on_remove_from_db_and_move(self, path: Path) -> None:
        """Remove one text file from the database and move it to 'deleted'."""
        if self.engine is None or not path.is_file():
            return

        file_name = path.name
        deleted_folder = Path(SEARCH_FOLDER).absolute() / "deleted"

        confirmation = wx.MessageBox(
            f"Remove '{file_name}' from the database and move the original file to:\n\n"
            f"{deleted_folder}\n\n"
            f"Source: {path}",
            "Delete indexed file",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        )
        if confirmation != wx.YES:
            return

        # Cancel any previous completed-progress reset before starting a new job.
        self._stop_index_reset_timer()
        self._update_index_progress(0.0, "Removing file")

        def work():
            assert self.engine is not None

            # Find the document ID corresponding to this path.
            doc_id = None
            for row in self.engine.list_documents():
                if Path(row["path"]).absolute() == path.absolute():
                    doc_id = int(row["id"])
                    break

            if doc_id is None:
                raise ValueError(f"File '{file_name}' not found in database.")

            # delete_document may perform database maintenance/reindexing and reports
            # progress through the same callback used by indexing.
            result = self.engine.delete_document(
                doc_id,
                path=path,
                progress_callback=self._queue_index_progress,
            )

            # Move the original file only after database removal succeeded.
            if path.exists():
                deleted_folder.mkdir(parents=True, exist_ok=True)
                destination = deleted_folder / path.name

                if destination.exists():
                    stem = destination.stem
                    suffix = destination.suffix
                    serial = 1
                    while True:
                        candidate = deleted_folder / f"{stem} ({serial}){suffix}"
                        if not candidate.exists():
                            destination = candidate
                            break
                        serial += 1

                shutil.move(str(path), str(destination))

            return result

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                self._update_index_progress(0.0, "Failed")
                self.SetStatusText("File removal failed.")
                wx.MessageBox(str(error), "Error", wx.OK | wx.ICON_ERROR)
                return

            self._document_match_counts.pop(str(path.absolute()), None)
            self._reset_stats()

            if result and isinstance(result, dict):
                if result.get("regex") is not None:
                    self._set_stats_row(self.regex_stats, result["regex"])
                if result.get("fuzzy") is not None:
                    self._set_stats_row(self.fuzzy_stats, result["fuzzy"])

            self._refresh_tree()
            self._folder_snapshot = self._folder_state()
            self._folder_view_snapshot = dict(self._folder_snapshot)

            # Keep 100% visible for one second after deletion/reindexing.
            self._update_index_progress(1.0, "Ready")
            self._schedule_index_progress_reset(delay_ms=1000)

            self.SetStatusText(f"Removed '{path.name}' and moved to 'deleted'.")

        self._run_background(work, done)


    def _on_remove_folder_tree_and_move(self, folder_path: Path) -> None:
        """Remove all .txt files below a folder from the database and move them to 'deleted'."""
        if self.engine is None or not folder_path.is_dir():
            return

        search_folder = Path(SEARCH_FOLDER).absolute()
        folder_path = folder_path.absolute()

        # The search-folder root and the hidden deleted tree are never removable.
        if (
            folder_path == search_folder
            or not self._inside(folder_path, search_folder)
            or self._is_deleted_path(folder_path, search_folder)
        ):
            return

        try:
            text_files = sorted(
                (
                    p.absolute()
                    for p in folder_path.rglob("*.txt")
                    if p.is_file() and not self._is_deleted_path(p, search_folder)
                ),
                key=lambda p: str(p).casefold(),
            )
        except OSError as exc:
            wx.MessageBox(
                f"Could not scan '{folder_path}': {exc}",
                "Folder removal",
                wx.OK | wx.ICON_ERROR,
            )
            return

        confirmation = wx.MessageBox(
            f"Remove the complete folder tree from the database and move all .txt files "
            f"below it to:\n\n"
            f"{search_folder / 'deleted' / folder_path.name}\n\n"
            f"Folder: {folder_path}\n"
            f"Text files found: {len(text_files)}",
            "Delete folder tree",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING,
        )
        if confirmation != wx.YES:
            return

        self._stop_index_reset_timer()
        self._update_index_progress(0.0, "Removing folder tree")

        def work():
            assert self.engine is not None

            indexed_rows = list(self.engine.list_documents())
            indexed_by_path = {
                Path(row["path"]).absolute(): int(row["id"])
                for row in indexed_rows
            }

            targets = [
                (path, indexed_by_path[path])
                for path in text_files
                if path in indexed_by_path
            ]

            last_result = None
            total = max(1, len(targets))

            for index, (file_path, doc_id) in enumerate(targets):
                def item_progress(stage: str, fraction: float, base=index, count=total):
                    # Reserve the full 0..1 range across all deleted documents.
                    overall = (base + max(0.0, min(1.0, float(fraction)))) / count
                    self._queue_index_progress(
                        f"Removing {file_path.name}: {stage}",
                        overall,
                    )

                last_result = self.engine.delete_document(
                    doc_id,
                    path=file_path,
                    progress_callback=item_progress,
                    move_source=False,
                    maintenance=False,
                )

            # Storage cleanup is scheduled at startup (every 20th launch), not
            # during ordinary document removal/indexing operations.

            # Move only .txt files; preserve their folder structure below the
            # selected folder under SEARCH_FOLDER/deleted/<selected-folder>/...
            deleted_root = search_folder / "deleted"
            destination_root = deleted_root / folder_path.name
            if destination_root.exists():
                serial = 1
                base_name = folder_path.name
                while True:
                    candidate = deleted_root / f"{base_name} ({serial})"
                    if not candidate.exists():
                        destination_root = candidate
                        break
                    serial += 1

            moved_count = 0
            for source in text_files:
                if not source.exists():
                    continue
                relative = source.relative_to(folder_path)
                destination = destination_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved_count += 1

            # Remove empty directories left behind by the moved text files.
            try:
                remaining_dirs = sorted(
                    (p for p in folder_path.rglob("*") if p.is_dir()),
                    key=lambda p: len(p.parts),
                    reverse=True,
                )
            except OSError:
                remaining_dirs = []

            for directory in remaining_dirs:
                try:
                    directory.rmdir()
                except OSError:
                    # Keep directories containing non-text files or anything
                    # else that should not be deleted.
                    pass

            try:
                folder_path.rmdir()
            except OSError:
                pass

            return {
                "result": last_result,
                "indexed_count": len(targets),
                "moved_count": moved_count,
                "found_count": len(text_files),
            }

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                self._update_index_progress(0.0, "Failed")
                self.SetStatusText("Folder removal failed.")
                wx.MessageBox(str(error), "Folder removal", wx.OK | wx.ICON_ERROR)
                return

            # Remove stale pending paths from this tree.
            self.pending_paths = {
                p for p in self.pending_paths
                if not self._inside(p.absolute(), folder_path)
            }

            self._document_match_counts = {
                p: count
                for p, count in self._document_match_counts.items()
                if not self._inside(Path(p).absolute(), folder_path)
            }
            self._reset_stats()

            self._refresh_tree()
            self._folder_snapshot = self._folder_state()
            self._folder_view_snapshot = dict(self._folder_snapshot)

            self._update_index_progress(1.0, "Ready")
            self._schedule_index_progress_reset(delay_ms=1000)

            found_count = int(result.get("found_count", 0)) if isinstance(result, dict) else 0
            moved_count = int(result.get("moved_count", 0)) if isinstance(result, dict) else 0
            indexed_count = int(result.get("indexed_count", 0)) if isinstance(result, dict) else 0
            self.SetStatusText(
                f"Removed folder tree '{folder_path.name}': "
                f"{indexed_count} database document(s) removed, "
                f"{moved_count}/{found_count} text file(s) moved to 'deleted'."
            )

        self._run_background(work, done)

    def _on_open_file(self, path: Path) -> None:
        """Open file with system default viewer."""
        try:
            wx.LaunchDefaultBrowser(str(path)) # This works for files on some platforms, but better:
            import subprocess
            if sys.platform == "win32":
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.call(["open", str(path)])
            else:
                subprocess.call(["xdg-open", str(path)])
        except Exception as e:
            wx.MessageBox(f"Could not open file: {e}", "Error", wx.OK | wx.ICON_ERROR)




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

    def _finish_search_indicator(self, label: str = "Ready", reset_delay_ms: int = 1000) -> None:
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

    def _stop_index_reset_timer(self) -> None:
        """Stop and clear the delayed index-progress reset timer safely."""
        timer = self._index_reset_timer
        self._index_reset_timer = None
        if timer is None:
            return
        try:
            if timer.IsRunning():
                timer.Stop()
        except AttributeError:
            # Be defensive for wx ports/types without IsRunning().
            try:
                timer.Stop()
            except Exception:
                pass

    def _schedule_index_progress_reset(self, delay_ms: int = 1000) -> None:
        """Show completed progress briefly, then reset the index gauge."""
        self._stop_index_reset_timer()
        self._index_reset_timer = wx.CallLater(
            delay_ms,
            self._reset_index_progress,
        )

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
            # These copied files are not indexed yet. Keep _folder_snapshot at
            # the last synchronized state so "Index new files" will detect them.
            self._folder_view_snapshot = self._folder_state()
            self._refresh_tree()
            copied_count = len(result or [])
            self.SetStatusText(
                f"Copied {copied_count} text file(s) to {folder}."
                if copied_count
                else "No .txt files were selected."
            )

        self._run_background(copy_files, done)

    def _on_choose_folder(self, _event: wx.CommandEvent) -> None:
        """Show a directory dialog; copy all .txt files (recursively) into SEARCH_FOLDER."""
        dialog = wx.DirDialog(
            self,
            "Choose folder to scan for text files",
            style=wx.DD_DEFAULT_STYLE | wx.DD_NEW_DIR_BUTTON,
        )
        try:
            if dialog.ShowModal() != wx.ID_OK:
                return
            chosen_dir = Path(dialog.GetPath()).absolute()
        finally:
            dialog.Destroy()

        folder = Path(SEARCH_FOLDER).absolute()

        def copy_files() -> list[Path]:
            folder.mkdir(parents=True, exist_ok=True)
            copied: list[Path] = []
            for source in sorted(chosen_dir.rglob("*.txt"), key=lambda p: str(p).casefold()):
                if not source.is_file():
                    continue
                if "deleted" in source.relative_to(chosen_dir).parts:
                    continue
                # Preserve sub-folder structure relative to the chosen directory.
                rel = source.relative_to(chosen_dir)
                destination = folder / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Keep an existing file with the same name intact.
                if destination.exists() and destination.absolute() != source.absolute():
                    stem = destination.stem
                    suffix = destination.suffix
                    serial = 1
                    while True:
                        candidate = destination.parent / f"{stem} ({serial}){suffix}"
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
            # Files inside SEARCH_FOLDER are already tracked via the folder scanner;
            # add them to pending_paths so they appear in the tree as well.
            self.pending_paths.update(result or [])
            # These copied files are not indexed yet. Keep _folder_snapshot at
            # the last synchronized state so "Index new files" will detect them.
            self._folder_view_snapshot = self._folder_state()
            self._refresh_tree()
            copied_count = len(result or [])
            self.SetStatusText(
                f"Copied {copied_count} text file(s) from '{chosen_dir.name}' into {folder}."
                if copied_count
                else "No .txt files found in the selected folder."
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
                move_btn = wx.Button(panel, label="Reset database, but move files to deleted folder")                
                delete_btn = wx.Button(panel, label="Delete ALL files and RESET database")
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

        # Clear any pending reset from previous operations.
        self._stop_index_reset_timer()
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
            self._refresh_tree()
            self._folder_snapshot = {}
            self._folder_view_snapshot = self._folder_state()


            self._update_index_progress(1.0, "Ready")
            
            # Keep 100% visible for one second, then reset.
            self._schedule_index_progress_reset(delay_ms=1000)

            action_label = "deleted" if action == "delete" else "moved to deleted"
            file_count = int(result.get("file_count", 0)) if isinstance(result, dict) else 0
            self.SetStatusText(
                f"Database reset complete: {file_count} text file(s) {action_label}."
            )

        self._run_background(work, done)





    def _on_index_new(self, _event: wx.CommandEvent) -> None:
        folder = Path(SEARCH_FOLDER).absolute()

        # A manual index operation is only needed when the folder contents/metadata
        # changed or there are explicitly pending external files to add.
        current_snapshot = self._folder_state()
        if current_snapshot == self._folder_snapshot and not self.pending_paths:
            self.SetStatusText("No new files for indexing.")
            return

        # Stop any existing reset timer. The timer is optional, so never
        # call IsRunning() on None.
        self._stop_index_reset_timer()
        self._update_index_progress(0.0, "Starting")

        def work():
            if self.engine is None:
                raise RuntimeError("Database is still loading.")

            # sync_search_folder() now scans the complete search-folder tree and
            # incrementally indexes only added/changed documents. External pending
            # files are the only paths that still need a separate index_files call.
            sync_callback = self._scaled_progress(self._queue_index_progress, 0.0, 0.90)
            index_callback = self._scaled_progress(self._queue_index_progress, 0.90, 1.0)

            sync_result = self.engine.sync_search_folder(progress_callback=sync_callback)

            external_pending = sorted(
                (
                    p.absolute()
                    for p in self.pending_paths
                    if not self._inside(p, folder)
                ),
                key=lambda p: str(p).casefold(),
            )
            manual_result = (
                self.engine.index_files(
                    external_pending,
                    progress_callback=index_callback,
                )
                if external_pending
                else {
                    "added": 0,
                    "changed": 0,
                    "removed": 0,
                    "unchanged": 0,
                }
            )

            return sync_result, manual_result

        def done(result, error) -> None:
            self._set_busy(False)
            if error:
                self._update_index_progress(0.0, "Failed")
                self.SetStatusText("Indexing failed.")
                wx.MessageBox(str(error), "Indexing error", wx.OK | wx.ICON_ERROR)
                return

            self.pending_paths.clear()
            self._refresh_tree()
            sync_result, manual_result = result
            added = sync_result["added"] + manual_result["added"]
            changed = sync_result["changed"] + manual_result["changed"]
            removed = sync_result["removed"] + manual_result["removed"]

            if added == 0 and changed == 0 and removed == 0:
                self._update_index_progress(1.0, "Ready")
                self._schedule_index_progress_reset(delay_ms=1000)
                self._folder_snapshot = self._folder_state()
                self._folder_view_snapshot = dict(self._folder_snapshot)
                self.SetStatusText("No new files for indexing.")
                return

            self._update_index_progress(1.0, "Ready")
            self._schedule_index_progress_reset(delay_ms=1000)

            self._folder_snapshot = self._folder_state()
            self._folder_view_snapshot = dict(self._folder_snapshot)
            self.SetStatusText(
                f"Index updated: {added} added, {changed} changed, {removed} removed."
            )

        self._run_background(work, done)



    def _on_document_browser(self, _event: wx.CommandEvent) -> None:
        """Open or foreground the modeless browser for the latest search state."""
        if not self._last_search_hits:
            wx.MessageBox(
                "Run a search with results first.",
                "Document Browser",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        if self._document_browser is not None:
            try:
                if not self._document_browser.IsBeingDeleted():
                    self._document_browser.update_search_state(self._last_search_hits)
                    self._document_browser.Raise()
                    self._document_browser.SetFocus()
                    return
            except RuntimeError:
                pass
            self._document_browser = None

        from document_browser import DocumentBrowserFrame

        self._document_browser = DocumentBrowserFrame(self, self._last_search_hits)
        self._document_browser.Show(True)
        self._document_browser.Raise()
        self._document_browser.SetFocus()

    def _on_search(self, _event: wx.CommandEvent) -> None:
        raw_terms = [
            self.keyword_1.GetValue(),
            self.keyword_2.GetValue(),
            self.keyword_3.GetValue(),
        ]
        raw_case_flags = [
            checkbox.GetValue() for checkbox in self._case_sensitive_fields
        ]
        term_entries = [
            (term.strip(), bool(case_sensitive))
            for term, case_sensitive in zip(raw_terms, raw_case_flags)
            if term.strip()
        ]
        terms = [term for term, _case_sensitive in term_entries]
        case_sensitive = [flag for _term, flag in term_entries]
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
            max_results_per_document = self._read_max_results_per_document(
                self.max_results_per_document
            )
        except ValueError as exc:
            wx.MessageBox(str(exc), "Search settings", wx.OK | wx.ICON_WARNING)
            return

        selection = self.mode.GetSelection()
        mode = ("regex", "fuzzy", "both")[selection]
        show_time_report = self.timelogs.GetValue()

        # Collect currently checked leaf paths from the tree.
        selected_paths: set[Path] = self._get_selected_leaf_paths()
        if not selected_paths:
            wx.MessageBox(
                "No text files are checked in the file list.\n"
                "Check at least one file or folder before searching.",
                "Search",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        # Check for unindexed documents
        if self.engine is not None:
            indexed_paths = {
                Path(row["path"]).absolute()
                for row in self.engine.list_documents()
            }
            unindexed = selected_paths - indexed_paths
            if unindexed:
                wx.MessageBox(
                    "One or more documents selected but not indexed.",
                    "Warning",
                    wx.OK | wx.ICON_WARNING,
                )
        # Continue with search as usual

        # START TIMING
        t_start = time.perf_counter()
        self._search_timing = {
            "start": t_start,
            "pre_search": 0.0,
            "search_engine": 0.0,
            "format_results": 0.0,
            "ui_update": 0.0,
            "total": 0.0,
        }

        self._start_search_indicator(mode, f"Searching {len(selected_paths)} selected file(s) …")

        def _filter_hits_by_selection(
            hits: list,
            paths: set[Path],
        ) -> list:
            """Keep a unique-content hit when any physical copy is selected.

            Search is indexed once per SHA-256, but the user selects physical
            file paths. Match against all locations and prefer a selected path
            for downstream actions such as opening the source file.
            """
            filtered: list = []
            normalized_selected = {Path(path).absolute() for path in paths}
            for hit in hits:
                document = hit.document
                raw_locations = document.get("locations") or []
                locations = [Path(str(value)).absolute() for value in raw_locations]
                if not locations:
                    fallback_path = str(document.get("path", ""))
                    if fallback_path:
                        locations = [Path(fallback_path).absolute()]
                selected_locations = sorted(
                    (path for path in locations if path in normalized_selected),
                    key=lambda path: str(path).casefold(),
                )
                if selected_locations:
                    document["selected_locations"] = [str(path) for path in selected_locations]
                    document["path"] = str(selected_locations[0])
                    filtered.append(hit)
            return filtered

        def run_search(
            search_mode: str,
            progress_callback,
        ) -> tuple[list, object, dict[str, object]]:
            if self.engine is None:
                raise RuntimeError("Database is still loading.")
            # Time the engine call specifically
            t_engine_start = time.perf_counter()
            hits, report = self.engine.search_with_report(
                terms,
                search_mode,
                fuzzy_distance=fuzzy_distance,
                max_distance_chars=max_distance_chars,
                snippet_context_chars=snippet_context_chars,
                max_results=max_results,
                max_results_per_document=max_results_per_document,
                case_sensitive=case_sensitive,
                progress_callback=progress_callback,
            )
            t_engine_end = time.perf_counter()
            
            # Update the timing dict (thread-safe enough for this use case as it's read in done)
            self._search_timing["search_engine"] += (t_engine_end - t_engine_start) * 1000.0
            
            hits = _filter_hits_by_selection(hits, selected_paths)
            return hits, report, self._stats_from_hits(hits, report)

        def work():
            t_work_start = time.perf_counter()
            
            if mode != "both":
                hits, report, stats = run_search(
                    mode,
                    self._queue_search_progress,
                )
                
                # Time formatting
                t_format_start = time.perf_counter()
                text = self._get_engine_module().format_results(
                    hits,
                    max_distance_chars=max_distance_chars,
                    snippet_context_chars=snippet_context_chars,
                    report=report if show_time_report else None,
                )
                t_format_end = time.perf_counter()
                
                self._search_timing["format_results"] += (t_format_end - t_format_start) * 1000.0
                
                return {
                    "text": text,
                    "regex": stats if mode == "regex" else None,
                    "fuzzy": stats if mode == "fuzzy" else None,
                    "hits": hits,
                }

            if mode == "both":
                regex_hits, regex_report, regex_stats = run_search(
                    "regex",
                    self._scaled_progress(self._queue_search_progress, 0.0, 0.50),
                )
                fuzzy_hits, fuzzy_report, fuzzy_stats = run_search(
                    "fuzzy",
                    self._scaled_progress(self._queue_search_progress, 0.50, 0.50),
                )
                engine_mod = self._get_engine_module()
                text = "\n".join([
                    "REGEX RESULTS",
                    "=" * 72,
                    engine_mod.format_results(
                        regex_hits,
                        max_distance_chars=max_distance_chars,
                        snippet_context_chars=snippet_context_chars,
                        report=regex_report if show_time_report else None,
                    ),
                    "",
                    "",
                    f"FUZZ RESULTS (edit distance <= {fuzzy_distance})",
                    "=" * 72,
                    engine_mod.format_results(
                        fuzzy_hits,
                        max_distance_chars=max_distance_chars,
                        snippet_context_chars=snippet_context_chars,
                        report=fuzzy_report if show_time_report else None,
                    ),
                ])
            
                return {
                    "text": text,
                    "regex": regex_stats,
                    "fuzzy": fuzzy_stats,
                    "hits": {"regex": regex_hits, "fuzzy": fuzzy_hits},
                }



        def done(result, error) -> None:
            t_done_start = time.perf_counter()
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

            # Update match counts per document
            self._document_match_counts.clear()
            
            hits_data = result.get("hits")
            if hits_data:
                if mode == "both":
                    all_hits = []
                    if isinstance(hits_data, dict):
                        all_hits.extend(hits_data.get("regex", []))
                        all_hits.extend(hits_data.get("fuzzy", []))
                    else:
                        all_hits = hits_data
                else:
                    all_hits = hits_data if isinstance(hits_data, list) else []
                
                # Count matches per document
                for hit in all_hits:
                    # One hit represents unique content, but all selected copies
                    # have the same matched text and therefore share its chain count.
                    matched_paths = hit.document.get("selected_locations") or [
                        hit.document.get("path", "")
                    ]
                    for doc_path in matched_paths:
                        doc_path = str(doc_path)
                        if doc_path:
                            self._document_match_counts[doc_path] = (
                                self._document_match_counts.get(doc_path, 0) + len(hit.chains)
                            )

            self._reset_stats()
            if result["regex"] is not None:
                self._set_stats_row(self.regex_stats, result["regex"])
            if result["fuzzy"] is not None:
                self._set_stats_row(self.fuzzy_stats, result["fuzzy"])
                
            # Time UI update
            t_ui_start = time.perf_counter()

            self._set_output_text_with_matches(
                result["text"] or "No proximity matches found."
            )
            self.output.SetCurrentPos(0)
            self.output.SetAnchor(0)
            self.output.ScrollToStart()
            self.output.SetFocus()
            t_ui_end = time.perf_counter()

            self._search_timing["ui_update"] += (t_ui_end - t_ui_start) * 1000.0
             
            self._finish_search_indicator("Search complete.", reset_delay_ms=1000)
            self.SetStatusText("Search complete.")
            
            # Refresh tree to show updated match counts
            self._update_tree_results()

            # Preserve the completed, selection-filtered search result state for
            # the modeless Document Browser. If it is already open, refresh it
            # immediately so it continues to represent the last search.
            self._last_search_hits = result.get("hits")
            self.btn_document_browser.Enable(bool(self._last_search_hits))
            if self._document_browser is not None:
                try:
                    if self._document_browser.IsBeingDeleted():
                        self._document_browser = None
                    else:
                        self._document_browser.update_search_state(self._last_search_hits)
                        self._document_browser.Raise()
                except RuntimeError:
                    self._document_browser = None

            # Calculate total and log
            t_done_end = time.perf_counter()
            total_ms = (t_done_end - t_start) * 1000.0
            self._search_timing["total"] = total_ms
            
            # Print detailed breakdown to console/debug log
            print(f"--- SEARCH TIMING BREAKDOWN ---")
            print(f"Total Time: {total_ms:.2f} ms")
            print(f"Engine Search: {self._search_timing['search_engine']:.2f} ms")
            print(f"Format Results: {self._search_timing['format_results']:.2f} ms")
            print(f"UI Update (SetText/Styling): {self._search_timing['ui_update']:.2f} ms")
            print(f"Other/Overhead: {total_ms - self._search_timing['search_engine'] - self._search_timing['format_results'] - self._search_timing['ui_update']:.2f} ms")
            print("---------------------------")


        self._run_background(work, done)


    @staticmethod
    def _inside(path: Path, folder: Path) -> bool:
        try:
            path.relative_to(folder)
            return True
        except ValueError:
            return False

    @staticmethod
    def _is_deleted_path(path: Path, folder: Path) -> bool:
        try:
            rel = path.relative_to(folder)
        except ValueError:
            return "deleted" in path.parts
        return "deleted" in rel.parts



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


