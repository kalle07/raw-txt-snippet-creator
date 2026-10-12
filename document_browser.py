from __future__ import annotations

"""Modeless document browser for the text-search GUI.

The browser consumes the public SearchHit shape without importing the search
engine. This keeps GUI startup lightweight and lets the main search window stay
fully usable while the browser is open.
"""

from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import os
import subprocess
import sys
import unicodedata
import wx
import wx.stc as stc


MATCH_STYLE = 1
CURRENT_MATCH_STYLE = 2
CHAIN_STYLE = 3


@dataclass(slots=True)
class BrowserDocument:
    """Normalized search-result data for one browsable document."""

    path: str
    filename: str
    content: str
    chains: list[list[dict[str, Any]]]
    occurrence_ranges: list[tuple[int, int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def result_count(self) -> int:
        """Return the number of distinct result chains in this document."""
        return len(self.chains)


# HELPER

# Maps control, format, and surrogate code points to U+0020 (space).
# Newline characters are deliberately excluded so they can be preserved.
_CONTROL_TO_SPACE = str.maketrans(
    {
        ch: " "
        for ch in (chr(i) for i in range(0x110000))
        if unicodedata.category(ch) in {"Cc", "Cf", "Cs"}
        and ch not in {"\n", "\r"}
    }
)

_NEWLINE_CHARS = "\n\r"


def sanitize_for_display(
    text: str,
    *,
    placeholder: str = " ",
    keep_newlines: bool = True,
    keep_tabs: bool = True,
) -> str:
    """Replace non-printable/control characters while preserving character count.

    By default, preserves LF (``\\n``), CR (``\\r``), and CRLF (``\\r\\n``)
    line endings, as well as tabs.
    """
    if not text:
        return text

    result = text.translate(_CONTROL_TO_SPACE)

    chars_to_replace = set()

    if not keep_newlines:
        chars_to_replace.update(_NEWLINE_CHARS)

    if not keep_tabs:
        chars_to_replace.add("\t")

    if chars_to_replace:
        result = result.translate(
            str.maketrans({ch: placeholder for ch in chars_to_replace})
        )

    return result

# minimap

class MatchMinimap(wx.Panel):
    """
    Compact document-height indicator with one mark per final result chain.

    This version uses a simple fixed vertical offset (_arrow_offset) so that
    the minimap's logical range matches the scrollbar thumb track (excluding
    the arrow buttons).
    """

    # Tweak this value until the markers align with the scrollbar track.
    # Typical values are 12–20 depending on theme / DPI.
    _ARROW_OFFSET = 20

    def __init__(
        self,
        parent: wx.Window,
        on_click: callable | None = None,
    ) -> None:
        super().__init__(parent, size=(11, -1))

        self.SetMinSize((11, -1))
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

        self._positions: list[int] = []
        self._line_count = 1
        self._current_position: int | None = None
        self._on_click_callback = on_click

        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)

    def SetMarkers(
        self,
        positions: Iterable[int],
        line_count: int,
        current_position: int | None = None,
    ) -> None:  # noqa: N802
        """Set occurrence lines and the active-result line."""
        self._positions = sorted(
            {
                max(0, int(position))
                for position in positions
            }
        )

        self._line_count = max(1, int(line_count))

        self._current_position = (
            None
            if current_position is None
            else max(0, int(current_position))
        )

        self.Refresh(False)

    def _y_for_line(self, line: int, height: int) -> int:
        """Map a zero-based document line into the minimap's usable track."""
        offset = self._ARROW_OFFSET

        # Ensure the track has at least 1 pixel.
        track_top = offset
        track_bottom = max(track_top + 1, height - offset - 1)
        track_height = max(1, track_bottom - track_top)

        if self._line_count <= 1:
            return track_top

        bounded_line = max(
            0,
            min(line, self._line_count - 1),
        )

        fraction = bounded_line / (self._line_count - 1)

        return track_top + int(round(fraction * track_height))

    def _fraction_for_y(self, y: int, height: int) -> float:
        """Map a minimap y coordinate to a normalized document fraction."""
        offset = self._ARROW_OFFSET

        track_top = offset
        track_bottom = max(track_top + 1, height - offset - 1)
        track_height = max(1, track_bottom - track_top)

        clamped_y = max(track_top, min(y, track_bottom))

        return (clamped_y - track_top) / float(track_height)

    def _on_paint(self, _event: wx.PaintEvent) -> None:
        """Paint minimap background, match markers, and active marker."""
        dc = wx.AutoBufferedPaintDC(self)

        width, height = self.GetClientSize()

        dc.SetBackground(wx.Brush(wx.Colour(170, 170, 170)))
        dc.Clear()

        if width <= 1 or height <= 1:
            return

        offset = self._ARROW_OFFSET
        track_top = offset
        track_bottom = max(track_top + 1, height - offset - 1)

        # Optional: shade the arrow regions to make it obvious they are excluded.
        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.SetBrush(wx.Brush(wx.Colour(145, 145, 145)))

        if track_top > 0:
            dc.DrawRectangle(0, 0, width, track_top)

        bottom_height = height - track_bottom - 1
        if bottom_height > 0:
            dc.DrawRectangle(0, track_bottom + 1, width, bottom_height)

        marker_colour = wx.Colour(28, 78, 140)

        dc.SetPen(wx.Pen(marker_colour))
        dc.SetBrush(wx.Brush(marker_colour))

        for line in self._positions:
            y = self._y_for_line(line, height)

            dc.DrawRectangle(
                1,
                max(track_top, y - 1),
                max(2, width - 2),
                3,
            )

        if self._current_position is not None:
            active_colour = wx.Colour(15, 45, 85)

            y = self._y_for_line(
                self._current_position,
                height,
            )

            dc.SetPen(wx.Pen(active_colour))
            dc.SetBrush(wx.Brush(active_colour))

            active_y = max(
                track_top,
                min(y - 2, track_bottom - 4),
            )

            dc.DrawRectangle(
                0,
                active_y,
                width,
                min(5, track_bottom - active_y + 1),
            )

    def _on_left_down(self, event: wx.MouseEvent) -> None:
        """Convert a minimap click into a normalized document position."""
        if self._on_click_callback is None:
            return

        _, height = self.GetClientSize()

        if height <= 1:
            return

        fraction = self._fraction_for_y(
            event.GetY(),
            height,
        )

        self._on_click_callback(fraction)
        

class DocumentBrowserFrame(wx.Frame):
    """Modeless ranked document browser with result navigation and minimap."""

    def __init__(
        self,
        parent: wx.Window,
        search_hits: Sequence[Any] | dict[str, Sequence[Any]] | None,
    ) -> None:
        super().__init__(
            parent,
            title="Document Browser",
            size=(1280, 900),
            style=wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT,
        )

        self._parent = parent
        self._documents: list[BrowserDocument] = []

        self._selected_index = -1
        self._result_index = -1
        self._loading_document = False
        self._destroying = False

        self._line_starts: list[int] = [0]
        self._line_start_byte_offsets: list[int] = [0]
        self._prefix_targets: dict[int, int] = {}
        self._range_byte_offsets: dict[
            tuple[int, int],
            tuple[int, int],
        ] = {}
        self._marker_positions: list[int] = []

        self._build_ui()
        self._bind_events()

        self.update_search_state(search_hits)

        self.CentreOnParent()
        self.Raise()
        self.SetFocus()

    # ------------------------------------------------------------------
    # Public integration API
    # ------------------------------------------------------------------

    def update_search_state(
        self,
        search_hits: Sequence[Any] | dict[str, Sequence[Any]] | None,
    ) -> None:
        """Replace browser content with the latest main-window search results."""
        previous_path: str | None = None

        if 0 <= self._selected_index < len(self._documents):
            previous_path = self._documents[self._selected_index].path

        self._documents = self._normalize_search_hits(search_hits)
        self._populate_document_list()

        if not self._documents:
            self._selected_index = -1
            self._clear_editor("No search results available.")
            self.Raise()
            return

        selected_index = next(
            (
                index
                for index, document in enumerate(self._documents)
                if document.path == previous_path
            ),
            0,
        )

        self._select_document(selected_index)
        self.Raise()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct the browser layout and controls."""
        root = wx.Panel(self)
        root_sizer = wx.BoxSizer(wx.VERTICAL)

        splitter = wx.SplitterWindow(
            root,
            style=wx.SP_LIVE_UPDATE,
        )

        upper_panel = wx.Panel(splitter)
        lower_panel = wx.Panel(splitter)

        self._build_document_list_panel(upper_panel)
        self._build_document_viewer_panel(lower_panel)

        splitter.SplitHorizontally(
            upper_panel,
            lower_panel,
            sashPosition=225,
        )

        root_sizer.Add(
            splitter,
            1,
            wx.EXPAND | wx.ALL,
            8,
        )

        root.SetSizer(root_sizer)

    def _build_document_list_panel(self, parent: wx.Panel) -> None:
        """Build the ranked-document list in the upper splitter pane."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        sizer.Add(
            wx.StaticText(parent, label="Ranked search results"),
            0,
            wx.ALL,
            8,
        )

        self.document_list = wx.ListCtrl(
            parent,
            style=wx.LC_REPORT | wx.LC_SINGLE_SEL | wx.BORDER_SUNKEN,
        )

        self.document_list.AppendColumn("Position", width=75)
        self.document_list.AppendColumn("Filename", width=620)
        self.document_list.AppendColumn("Results", width=90)

        sizer.Add(
            self.document_list,
            1,
            wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
            8,
        )

        parent.SetSizer(sizer)

    def _build_document_viewer_panel(self, parent: wx.Panel) -> None:
        """Build result controls, read-only editor, and clickable minimap."""
        sizer = wx.BoxSizer(wx.VERTICAL)

        navigation_row = wx.BoxSizer(wx.HORIZONTAL)
        navigation_row.AddStretchSpacer(1)

        self.result_number = wx.TextCtrl(
            parent,
            size=(55, -1),
            style=wx.TE_PROCESS_ENTER,
        )
        self.result_number.SetToolTip(
            "Enter a result number, then press Enter or click Go."
        )

        navigation_row.Add(
            self.result_number,
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
        )

        self.btn_go = wx.Button(
            parent,
            label="Go",
            size=(36, 26),
        )
        self.btn_go.SetToolTip("Jump to the entered result number")

        navigation_row.Add(
            self.btn_go,
            0,
            wx.RIGHT,
            12,
        )

        self.result_label = wx.StaticText(
            parent,
            label="Results: 0 / 0",
        )

        navigation_row.Add(
            self.result_label,
            0,
            wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            8,
        )

        self.btn_previous = wx.Button(
            parent,
            label="◀",
            size=(32, 26),
        )
        self.btn_previous.SetToolTip("Previous result")
        self.btn_previous.Enable(False)

        navigation_row.Add(
            self.btn_previous,
            0,
            wx.RIGHT,
            3,
        )

        self.btn_next = wx.Button(
            parent,
            label="▶",
            size=(32, 26),
        )
        self.btn_next.SetToolTip("Next result")
        self.btn_next.Enable(False)

        navigation_row.Add(self.btn_next, 0)

        sizer.Add(
            navigation_row,
            0,
            wx.EXPAND | wx.ALL,
            8,
        )

        editor_row = wx.BoxSizer(wx.HORIZONTAL)

        self.editor = stc.StyledTextCtrl(
            parent,
            style=wx.BORDER_SUNKEN | wx.CLIP_CHILDREN,
        )

        self._configure_editor()

        editor_row.Add(
            self.editor,
            1,
            wx.EXPAND,
        )

        self.minimap = MatchMinimap(
            parent,
            on_click=self._on_minimap_click,
        )

        editor_row.Add(
            self.minimap,
            0,
            wx.EXPAND | wx.LEFT,
            3,
        )

        sizer.Add(
            editor_row,
            1,
            wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM,
            8,
        )

        parent.SetSizer(sizer)

    def _configure_editor(self) -> None:
        """Configure the StyledTextCtrl as a read-only document viewer."""
        self.editor.SetLexer(stc.STC_LEX_NULL)
        self.editor.SetReadOnly(True)

        self.editor.SetTabWidth(4)
        self.editor.SetUseTabs(False)

        #self.editor.SetWrapMode(stc.STC_WRAP_NONE)
        self.editor.SetWrapMode(stc.STC_WRAP_WORD)

        # Use Scintilla's own vertical scrollbar. No external scrollbar exists.
        self.editor.SetUseVerticalScrollBar(True)
        self.editor.SetUseHorizontalScrollBar(False)

        self.editor.SetMarginWidth(0, 42)
        self.editor.SetMarginType(0, stc.STC_MARGIN_NUMBER)

        self.editor.SetMarginWidth(1, 0)
        self.editor.SetMarginWidth(2, 0)

        self.editor.StyleSetBackground(
            MATCH_STYLE,
            wx.Colour(255, 235, 140),
        )
        self.editor.StyleSetForeground(
            MATCH_STYLE,
            wx.BLACK,
        )

        self.editor.StyleSetBackground(
            CURRENT_MATCH_STYLE,
            wx.Colour(255, 190, 80),
        )
        self.editor.StyleSetForeground(
            CURRENT_MATCH_STYLE,
            wx.BLACK,
        )

        self.editor.StyleSetBackground(
           CHAIN_STYLE,
           wx.Colour(158, 242, 158),
        )
        self.editor.StyleSetForeground(
           CHAIN_STYLE,
           wx.BLACK,
        )


    def _bind_events(self) -> None:
        """Bind only application events; no STC update or external-scroll events."""
        self.document_list.Bind(
            wx.EVT_LIST_ITEM_SELECTED,
            self._on_document_selected,
        )
        self.document_list.Bind(
            wx.EVT_LIST_ITEM_RIGHT_CLICK,
            self._on_document_context_menu,
        )

        self.btn_previous.Bind(
            wx.EVT_BUTTON,
            self._on_previous_result,
        )

        self.btn_next.Bind(
            wx.EVT_BUTTON,
            self._on_next_result,
        )

        self.result_number.Bind(
            wx.EVT_TEXT_ENTER,
            self._on_result_number_enter,
        )

        self.btn_go.Bind(
            wx.EVT_BUTTON,
            self._on_go_result,
        )

        self.Bind(wx.EVT_CLOSE, self._on_close)


    # ------------------------------------------------------------------
    # Document-list context menu actions
    # ------------------------------------------------------------------

    def _on_document_context_menu(self, event: wx.ListEvent) -> None:
        """Show actions for the document row that was right-clicked."""
        index = int(event.GetIndex())
        if not (0 <= index < len(self._documents)):
            return

        # A context-click should act on the row under the pointer, not on a
        # previously selected row.
        self._select_document(index)

        menu = wx.Menu()
        save_item = menu.Append(wx.ID_ANY, "Save chains")
        open_source_item = menu.Append(wx.ID_ANY, "Open source path")
        open_file_item = menu.Append(wx.ID_ANY, "Open file")

        # Menu commands must be bound to the window that owns PopupMenu.
        # Binding directly to wx.Menu can leave commands unhandled on some
        # wxPython/platform combinations, which looks like a silent no-op.
        save_handler = lambda _evt, row=index: self._save_chains(row)
        open_source_handler = lambda _evt, row=index: self._open_source_path(row)
        open_file_handler = lambda _evt, row=index: self._open_document_file(row)
        menu_bindings = (
            (save_item.GetId(), save_handler),
            (open_source_item.GetId(), open_source_handler),
            (open_file_item.GetId(), open_file_handler),
        )

        for item_id, handler in menu_bindings:
            self.document_list.Bind(wx.EVT_MENU, handler, id=item_id)

        try:
            click_position = self.document_list.ScreenToClient(wx.GetMousePosition())
            self.document_list.PopupMenu(menu, click_position)
        finally:
            for item_id, handler in menu_bindings:
                self.document_list.Unbind(
                    wx.EVT_MENU,
                    id=item_id,
                    handler=handler,
                )
            menu.Destroy()

    def _save_chains(self, index: int) -> None:
        """Save document metadata and every chain to a text report."""
        if not (0 <= index < len(self._documents)):
            return

        document = self._documents[index]
        source_path = Path(document.path).expanduser()
        default_dir = source_path.parent
        if not default_dir.is_dir():
            default_dir = Path.cwd()

        default_name = f"report-{source_path.stem or document.filename}.txt"
        dialog = wx.FileDialog(
            self,
            message=f"Save chains for {document.filename}",
            defaultDir=str(default_dir),
            defaultFile=default_name,
            wildcard="Text files (*.txt)|*.txt|All files (*.*)|*.*",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        try:
            if dialog.ShowModal() != wx.ID_OK:
                return
            report_path = Path(dialog.GetPath())
        finally:
            dialog.Destroy()

        try:
            report_path.write_text(
                self._build_chain_report(document),
                encoding="utf-8",
                newline="\n",
            )
        except OSError as exc:
            wx.MessageBox(
                f"Could not save the chain report:\n{exc}",
                "Save chains failed",
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return

        wx.MessageBox(
            f"Chain report saved to:\n{report_path}",
            "Chains saved",
            wx.OK | wx.ICON_INFORMATION,
            self,
        )

    @staticmethod
    def _build_chain_report(document: BrowserDocument) -> str:
        """Format metadata and one complete text excerpt per result chain."""
        lines = [
            f"Document: {document.filename}",
            f"Source path: {document.path}",
            f"Result chains: {document.result_count}",
            "",
            "Document metadata:",
        ]

        if document.metadata:
            for key in sorted(document.metadata):
                value = document.metadata[key]
                # Keep metadata readable and prevent embedded newlines from
                # making one field look like multiple metadata entries.
                lines.append(f"  {key}: {value!r}")
        else:
            lines.append("  (No additional document metadata available.)")

        lines.extend(("", "Found chains:", ""))
        if not document.chains:
            lines.append("(No chains found.)")
            return "\n".join(lines) + "\n"

        for chain_number, chain in enumerate(document.chains, start=1):
            valid_positions: list[tuple[int, int]] = []
            for item in chain:
                try:
                    start = int(item.get("start", 0))
                    end = int(item.get("end", start))
                except (AttributeError, TypeError, ValueError):
                    continue
                if 0 <= start <= end <= len(document.content):
                    valid_positions.append((start, end))

            if not valid_positions:
                lines.append(f"Chain {chain_number} [position unavailable]: (text unavailable)")
                continue

            # Save the entire source span covered by the chain, not one line
            # per term. The reported position points to the first term only.
            chain_start = min(start for start, _end in valid_positions)
            chain_end = max(end for _start, end in valid_positions)
            line_number = document.content.count("\n", 0, chain_start) + 1
            previous_newline = document.content.rfind("\n", 0, chain_start)
            column_number = chain_start - previous_newline
            position = f"[term at char {chain_start + 1}; line {line_number}, column {column_number}]"
            chain_text = document.content[chain_start:chain_end]
            lines.append(f"Chain {chain_number} {position}: {chain_text!r}")

        return "\n".join(lines).rstrip() + "\n"

    def _open_source_path(self, index: int) -> None:
        """Open the source document's containing folder in the file manager."""
        if not (0 <= index < len(self._documents)):
            return
        source_path = Path(self._documents[index].path).expanduser()
        folder = source_path.parent
        if not folder.is_dir():
            wx.MessageBox(
                f"The source folder does not exist:\n{folder}",
                "Open source path failed",
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return
        self._launch_with_standard_app(folder)

    def _open_document_file(self, index: int) -> None:
        """Open the source document with the operating system's default app."""
        if not (0 <= index < len(self._documents)):
            return
        source_path = Path(self._documents[index].path).expanduser()
        if not source_path.is_file():
            wx.MessageBox(
                f"The source file does not exist:\n{source_path}",
                "Open file failed",
                wx.OK | wx.ICON_ERROR,
                self,
            )
            return
        self._launch_with_standard_app(source_path)

    def _launch_with_standard_app(self, path: Path) -> None:
        """Open a file or folder with the operating system's default handler."""
        try:
            # wx's launcher reports whether the request was accepted. Keep
            # platform-specific launchers as fallbacks for wx builds where it
            # cannot launch the target.
            if wx.LaunchDefaultApplication(str(path)):
                return

            if sys.platform.startswith("win"):
                os.startfile(str(path), "open")  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            wx.MessageBox(
                f"Could not open with the default application:\n{path}\n\n{exc}",
                "Open failed",
                wx.OK | wx.ICON_ERROR,
                self,
            )

    # ------------------------------------------------------------------
    # Search-hit normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_hits(
        search_hits: Sequence[Any] | dict[str, Sequence[Any]] | None,
    ) -> list[Any]:
        """Flatten supported search-hit containers into one ordered list."""
        if not search_hits:
            return []

        if isinstance(search_hits, dict):
            combined: list[Any] = []

            for key in ("regex", "fuzzy"):
                hits = search_hits.get(key, [])

                if hits:
                    combined.extend(hits)

            return combined

        return list(search_hits)

    @classmethod
    def _normalize_search_hits(
        cls,
        search_hits: Sequence[Any] | dict[str, Sequence[Any]] | None,
    ) -> list[BrowserDocument]:
        """Group hits by path, preserve match data, and sort display documents."""
        grouped: dict[str, BrowserDocument] = {}
        first_seen_order: list[str] = []

        for hit in cls._iter_hits(search_hits):
            source_document = getattr(hit, "document", {}) or {}

            path = str(source_document.get("path", ""))

            if not path:
                continue

            content = str(source_document.get("content", "") or "")

            browser_document = grouped.get(path)

            if browser_document is None:
                browser_document = BrowserDocument(
                    path=path,
                    filename=Path(path).name,
                    content=content,
                    chains=[],
                    occurrence_ranges=[],
                    metadata={
                        str(key): value
                        for key, value in source_document.items()
                        if key not in {"path", "content"}
                    } if isinstance(source_document, dict) else {},
                )

                grouped[path] = browser_document
                first_seen_order.append(path)

            browser_document.chains.extend(
                getattr(hit, "chains", []) or []
            )

            for term_matches in getattr(
                hit,
                "matches_by_term",
                [],
            ) or []:
                for match in term_matches or []:
                    try:
                        start = int(match["start"])
                        end = int(match["end"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    if 0 <= start < end <= len(browser_document.content):
                        browser_document.occurrence_ranges.append(
                            (start, end)
                        )

        for document in grouped.values():
            cls._deduplicate_document_matches(document)

        rank_order = {
            path: index
            for index, path in enumerate(first_seen_order)
        }

        documents = [
            grouped[path]
            for path in first_seen_order
            if grouped[path].result_count
        ]

        documents.sort(
            key=lambda document: (
                -document.result_count,
                rank_order[document.path],
            )
        )

        return documents

    @staticmethod
    def _deduplicate_document_matches(document: BrowserDocument) -> None:
        """Deduplicate chains and occurrence ranges while preserving useful order."""
        unique_chains: dict[
            tuple[tuple[int, int], ...],
            list[dict[str, Any]],
        ] = {}

        for chain in document.chains:
            signature = tuple(
                sorted(
                    (
                        int(item.get("start", 0)),
                        int(item.get("end", 0)),
                    )
                    for item in chain
                )
            )

            if signature and signature not in unique_chains:
                unique_chains[signature] = chain

        document.chains = list(unique_chains.values())

        document.chains.sort(
            key=lambda chain: (
                -len(chain),
                min(
                    (
                        int(item.get("start", 0))
                        for item in chain
                    ),
                    default=0,
                ),
            )
        )

        document.occurrence_ranges = sorted(
            set(document.occurrence_ranges)
        )

    # ------------------------------------------------------------------
    # Document list and selection
    # ------------------------------------------------------------------

    def _populate_document_list(self) -> None:
        """Render the current normalized document collection."""
        self.document_list.Freeze()

        try:
            self.document_list.DeleteAllItems()

            for position, document in enumerate(
                self._documents,
                start=1,
            ):
                row = self.document_list.InsertItem(
                    self.document_list.GetItemCount(),
                    str(position),
                )

                self.document_list.SetItem(
                    row,
                    1,
                    document.filename,
                )

                self.document_list.SetItem(
                    row,
                    2,
                    str(document.result_count),
                )
        finally:
            self.document_list.Thaw()

    def _on_document_selected(self, event: wx.ListEvent) -> None:
        """Load a document selected by the user without re-selecting its row."""
        index = int(event.GetIndex())

        if self._destroying:
            return

        if self._loading_document:
            return

        if index == self._selected_index:
            return

        if not (0 <= index < len(self._documents)):
            return

        self._selected_index = index
        self._load_document(self._documents[index])

    def _select_document(self, index: int) -> None:
        """Select and load a document for initial or programmatic selection."""
        if not (0 <= index < len(self._documents)):
            self._selected_index = -1
            self._clear_editor("No document selected.")
            return

        self._selected_index = index

        # Do not rely on the generated selection event to perform loading.
        # Loading happens explicitly below, protected by _loading_document.
        self._loading_document = True

        try:
            self.document_list.Select(index)
            self.document_list.Focus(index)
            self._load_document(self._documents[index])
        finally:
            self._loading_document = False

    # ------------------------------------------------------------------
    # Editor content and match highlighting
    # ------------------------------------------------------------------

    def _clear_editor(self, message: str) -> None:
        """Replace document content with an informational message."""
        self._result_index = -1
        self._line_starts = [0]
        self._line_start_byte_offsets = [0]
        self._prefix_targets = {}
        self._range_byte_offsets = {}
        self._marker_positions = []

        self.editor.SetReadOnly(False)

        try:
            self.editor.SetText(message)
            self.editor.EmptyUndoBuffer()

            self.editor.StartStyling(0)
            self.editor.SetStyling(
                self.editor.GetTextLength(),
                0,
            )
        finally:
            self.editor.SetReadOnly(True)

        self._update_navigation_controls()

        self.minimap.SetMarkers([], 1)

    @staticmethod
    def _line_starts_for_text(text: str) -> list[int]:
        """Return Python-character offsets at which each line begins."""
        starts = [0]

        for index, char in enumerate(text):
            if char == "\n":
                starts.append(index + 1)

        return starts

    @staticmethod
    def _char_to_utf8_offsets(
        text: str,
        targets: Iterable[int],
    ) -> dict[int, int]:
        """Map selected Python-character positions to UTF-8 byte offsets."""
        target_set = {
            max(0, min(len(text), int(position)))
            for position in targets
        }

        byte_offsets: dict[int, int] = {}
        byte_position = 0

        for char_index, char in enumerate(text):
            if char_index in target_set:
                byte_offsets[char_index] = byte_position

            byte_position += len(char.encode("utf-8"))

        if len(text) in target_set:
            byte_offsets[len(text)] = byte_position

        return byte_offsets

    @staticmethod
    def _chain_span(chain: list[dict[str, Any]]) -> tuple[int, int]:
       """Return the inclusive character span covering an entire chain."""
       starts: list[int] = []
       ends: list[int] = []

       for item in chain:
           try:
               start = int(item.get("start", 0))
               end = int(item.get("end", start))
           except (TypeError, ValueError):
               continue

           if end < start:
               continue

           starts.append(start)
           ends.append(end)

       if not starts:
           return 0, 0

       return min(starts), max(ends)


    def _load_document(self, document: BrowserDocument) -> None:
        """Load one document, apply base highlighting, and show its first result."""
        if self._destroying:
            return

        #text = document.content
        original_text = document.content

        text = sanitize_for_display(
            original_text,
            placeholder=" ",
            keep_newlines=True,
            keep_tabs=True,
        )

        self._line_starts = self._line_starts_for_text(text)
        self._range_byte_offsets = {}

        target_positions = self._collect_byte_offset_targets(document)
        target_positions.extend(self._line_starts)

        self._prefix_targets = self._char_to_utf8_offsets(
            text,
            target_positions,
        )

        self._line_start_byte_offsets = [
            self._prefix_targets[start]
            for start in self._line_starts
        ]

        for start, end in document.occurrence_ranges:
            start_byte = self._prefix_targets.get(start)
            end_byte = self._prefix_targets.get(end)

            if (
                start_byte is not None
                and end_byte is not None
                and end_byte > start_byte
            ):
                self._range_byte_offsets[(start, end)] = (
                    start_byte,
                    end_byte,
                )

        self.editor.SetReadOnly(False)

        try:
            self.editor.SetText(text)
            self.editor.EmptyUndoBuffer()
            self._apply_base_highlighting(document)
        finally:
            self.editor.SetReadOnly(True)

        self._result_index = 0 if document.chains else -1

        self.editor.ScrollToLine(0)
        self.editor.SetCurrentPos(0)
        self.editor.SetAnchor(0)

        self._marker_positions = self._marker_lines_for_document(document)

        current_line = self._current_result_line(document)

        self.minimap.SetMarkers(
            self._marker_positions,
            len(self._line_starts),
            current_line,
        )

        self._update_navigation_controls()

        if self._result_index >= 0:
            self._jump_to_result(update_index=False)

    def _collect_byte_offset_targets(
        self,
        document: BrowserDocument,
    ) -> list[int]:
        """Collect all character positions needed for styles and navigation."""
        targets: list[int] = []

        for start, end in document.occurrence_ranges:
            targets.extend((start, end))

        for chain in document.chains:
            for item in chain:
                try:
                    targets.append(int(item.get("start", 0)))
                    targets.append(int(item.get("end", 0)))
                except (TypeError, ValueError):
                    continue

        return targets

    def _apply_base_highlighting(
        self,
        document: BrowserDocument,
    ) -> None:
        """Clear styles, color chain spans subtly, and color every match yellow."""
        self.editor.StartStyling(0)
        self.editor.SetStyling(
            self.editor.GetTextLength(),
            0,
        )

        # Subtle background for every full chain span.
        for chain in document.chains:
            chain_start, chain_end = self._chain_span(chain)
            chain_byte_start = self._prefix_targets.get(chain_start)
            chain_byte_end = self._prefix_targets.get(chain_end)

            if (
                chain_byte_start is None
                or chain_byte_end is None
                or chain_byte_end <= chain_byte_start
            ):
                continue

            self.editor.StartStyling(chain_byte_start)
            self.editor.SetStyling(
                chain_byte_end - chain_byte_start,
                CHAIN_STYLE,
            )

        # Individual match highlights.
        for start, end in document.occurrence_ranges:
            byte_range = self._range_byte_offsets.get((start, end))

            if byte_range is None:
                continue

            byte_start, byte_end = byte_range

            self.editor.StartStyling(byte_start)
            self.editor.SetStyling(
                byte_end - byte_start,
                MATCH_STYLE,
            )



    def _highlight_current_chain(self) -> None:
        """Repaint base matches, subtle chain span, and overlay the active chain in orange."""
        document = self._current_document

        if document is None:
            return

        self.editor.SetReadOnly(False)

        try:
            self._apply_base_highlighting(document)

            if not (0 <= self._result_index < document.result_count):
                return

            chain = document.chains[self._result_index]

            # Subtle background for the entire span of the active chain.
            chain_start, chain_end = self._chain_span(chain)
            chain_byte_start = self._prefix_targets.get(chain_start)
            chain_byte_end = self._prefix_targets.get(chain_end)

            if (
                chain_byte_start is not None
                and chain_byte_end is not None
                and chain_byte_end > chain_byte_start
            ):
                self.editor.StartStyling(chain_byte_start)
                self.editor.SetStyling(
                    chain_byte_end - chain_byte_start,
                    CHAIN_STYLE,
                )

            # Stronger highlight for the individual matches inside the active chain.
            for item in chain:
                try:
                    start = int(item.get("start", 0))
                    end = int(item.get("end", start))
                except (TypeError, ValueError):
                    continue

                byte_start = self._prefix_targets.get(start)
                byte_end = self._prefix_targets.get(end)

                if (
                    byte_start is None
                    or byte_end is None
                    or byte_end <= byte_start
                ):
                    continue

                self.editor.StartStyling(byte_start)
                self.editor.SetStyling(
                    byte_end - byte_start,
                    CURRENT_MATCH_STYLE,
                )
        finally:
            self.editor.SetReadOnly(True)


    def _marker_lines_for_document(
        self,
        document: BrowserDocument,
    ) -> list[int]:
        """Return one document line per final result chain.

        Each chain is represented by the line containing its earliest match.
        This keeps the minimap aligned with result navigation instead of
        showing every individual term occurrence.
        """
        marker_lines: list[int] = []

        for chain in document.chains:
            if not chain:
                continue

            try:
                start = min(int(item.get("start", 0)) for item in chain)
            except (TypeError, ValueError):
                continue

            marker_lines.append(self._line_for_char_offset(start))

        return marker_lines

    def _current_result_line(
        self,
        document: BrowserDocument,
    ) -> int | None:
        """Return the first line occupied by the selected result chain."""
        if not (0 <= self._result_index < document.result_count):
            return None

        chain = document.chains[self._result_index]

        if not chain:
            return None

        start = min(
            int(item.get("start", 0))
            for item in chain
        )

        return self._line_for_char_offset(start)

    def _line_for_char_offset(self, offset: int) -> int:
        """Find the zero-based text line containing a character offset."""
        return max(
            0,
            bisect_right(self._line_starts, offset) - 1,
        )

    @property
    def _current_document(self) -> BrowserDocument | None:
        """Return the currently selected document, if one exists."""
        if 0 <= self._selected_index < len(self._documents):
            return self._documents[self._selected_index]

        return None

    # ------------------------------------------------------------------
    # Result navigation
    # ------------------------------------------------------------------

    def _update_navigation_controls(self) -> None:
        """Update the result count text and previous/next button state."""
        document = self._current_document

        count = document.result_count if document is not None else 0

        shown = (
            self._result_index + 1
            if 0 <= self._result_index < count
            else 0
        )

        self.result_label.SetLabel(
            f"Results: {shown} / {count}"
        )

        self.result_number.SetValue(str(shown))

        navigation_enabled = count > 0

        self.btn_previous.Enable(navigation_enabled)
        self.btn_next.Enable(navigation_enabled)


    def _on_previous_result(self, _event: wx.CommandEvent) -> None:
        """Move to the preceding result, wrapping at the first result."""
        document = self._current_document

        if document is None or not document.chains:
            return

        self._result_index = (
            self._result_index - 1
        ) % len(document.chains)

        self._jump_to_result()

    def _on_next_result(self, _event: wx.CommandEvent) -> None:
        """Move to the next result, wrapping at the final result."""
        document = self._current_document

        if document is None or not document.chains:
            return

        self._result_index = (
            self._result_index + 1
        ) % len(document.chains)

        self._jump_to_result()


    def _on_result_number_enter(self, _event: wx.TextEvent) -> None:
        """Jump when the user presses Enter in the result number field."""
        self._on_go_result(_event)

    def _on_go_result(self, _event: wx.CommandEvent) -> None:
        """Jump to the result number entered by the user."""
        document = self._current_document

        if document is None or not document.chains:
            return

        raw = self.result_number.GetValue().strip()

        try:
            requested = int(raw)
        except ValueError:
            self.result_number.SetValue(
                str(self._result_index + 1)
                if 0 <= self._result_index < document.result_count
                else "0"
            )
            self.result_number.SetSelection(0, -1)
            self.result_number.SetFocus()
            return

        if 1 <= requested <= document.result_count:
            self._result_index = requested - 1
            self._jump_to_result(update_index=False)
        else:
            self.result_number.SetValue(
                str(self._result_index + 1)
                if 0 <= self._result_index < document.result_count
                else "0"
            )
            self.result_number.SetSelection(0, -1)
            self.result_number.SetFocus()



    def _jump_to_result(self, update_index: bool = True) -> None:
        """Scroll to the active chain and refresh active-chain highlighting."""
        document = self._current_document

        if document is None or not document.chains:
            self._update_navigation_controls()
            return

        if update_index:
            self._result_index %= len(document.chains)

        chain = document.chains[self._result_index]

        if not chain:
            self._update_navigation_controls()
            return

        start = min(
            int(item.get("start", 0))
            for item in chain
        )

        byte_position = self._prefix_targets.get(start, 0)

        self.editor.GotoPos(byte_position)
        self.editor.SetCurrentPos(byte_position)
        self.editor.SetAnchor(byte_position)
        self.editor.EnsureCaretVisible()

        self._highlight_current_chain()

        self.minimap.SetMarkers(
            self._marker_positions,
            len(self._line_starts),
            self._line_for_char_offset(start),
        )

        self._update_navigation_controls()

    # ------------------------------------------------------------------
    # Simple minimap
    # ------------------------------------------------------------------

    def _on_minimap_click(self, fraction: float) -> None:
        """Scroll to the clicked proportional logical document position."""
        if self._destroying:
            return

        total_lines = max(1, len(self._line_starts))

        target_line = int(
            round(
                fraction * max(0, total_lines - 1)
            )
        )

        if 0 <= target_line < len(self._line_starts):
            byte_position = self._line_start_byte_offsets[target_line]
            self.editor.GotoPos(byte_position)
            self.editor.EnsureCaretVisible()


    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def _on_close(self, event: wx.CloseEvent) -> None:
        """Mark shutdown, then let wxPython complete normal destruction."""
        self._destroying = True
        event.Skip()


__all__ = ["BrowserDocument", "DocumentBrowserFrame"]