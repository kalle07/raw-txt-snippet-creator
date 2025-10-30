import wx
import os
import re
import json
import threading
from pathlib import Path
from rapidfuzz import fuzz, process
from typing import List, Tuple, Any, Optional


# --- Engine ---

class TextProcessor:
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.text_bytes = None
        self.decoded_text = None
        self.char_to_byte = None
        self.load_and_process_file()

    def load_and_process_file(self):
        try:
            self.text_bytes = self.file_path.read_bytes()
            self.decoded_text = self.text_bytes.decode("utf-8", errors="surrogateescape")
            self._build_char_to_byte_mapping()
        except Exception as e:
            raise RuntimeError(f"Failed to read file {self.file_path}: {str(e)}")

    def _build_char_to_byte_mapping(self):
        self.char_to_byte = [0]
        for ch in self.decoded_text:
            self.char_to_byte.append(self.char_to_byte[-1] + len(ch.encode("utf-8", errors="surrogateescape")))


class Match:
    def __init__(self, pattern, text, start_char, end_char):
        self.pattern = pattern
        self.text = text
        self.start_char = start_char
        self.end_char = end_char
        self.byte_start = None
        self.byte_end = None

    def set_byte_positions(self, char_to_byte_map):
        self.byte_start = char_to_byte_map[self.start_char]
        self.byte_end = char_to_byte_map[self.end_char]


class SnippetExtractor:
    # Pre-compiled regex patterns for performance
    _regex_cache = {}

    # -------------
    # wildcard part
    # -------------
    @staticmethod
    def wildcards_to_regex(pattern: str) -> str:
        """
        Convert wildcard pattern to regex with caching.
        - '?'  → matches exactly one character of any type
        - '*'  → matches zero or more non-whitespace chars
        """
        try:
            # Use cache for better performance
            if pattern in SnippetExtractor._regex_cache:
                return SnippetExtractor._regex_cache[pattern]
            
            regex_parts = []
            i = 0
            while i < len(pattern):
                ch = pattern[i]
                if ch == '?':
                    regex_parts.append('.')
                    i += 1
                elif ch == '*':
                    regex_parts.append(r'(?:\S*)')
                    i += 1
                else:
                    regex_parts.append(re.escape(ch))
                    i += 1

            result = "".join(regex_parts)
            SnippetExtractor._regex_cache[pattern] = result
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to convert wildcard pattern '{pattern}' to regex: {str(e)}")


    @staticmethod
    # hanlde wildcard pattern '?' and '*'
    def expand_to_word_boundaries(text: str, start_char: int, end_char: int, pattern: str):
        """
        Expand match boundaries depending on '*' position.
        """
        try:
            # Exact match for '?' only patterns
            if '?' in pattern and '*' not in pattern:
                return text[start_char:end_char], start_char, end_char

            expanded_start = start_char
            expanded_end = end_char

            if '*' in pattern:
                if pattern.startswith('*') and not pattern.endswith('*'):
                    # expand LEFT until whitespace - optimized with backward search
                    while expanded_start > 0 and not text[expanded_start - 1].isspace():
                        expanded_start -= 1

                elif pattern.endswith('*') and not pattern.startswith('*'):
                    # expand RIGHT until whitespace - optimized forward search
                    while expanded_end < len(text) and not text[expanded_end].isspace():
                        expanded_end += 1

                else:
                    # '*' is inside → expand both sides until visible character
                    if expanded_start > 0:
                        expanded_start -= 1
                    if expanded_end < len(text):
                        expanded_end += 1

            return text[expanded_start:expanded_end], expanded_start, expanded_end

        except Exception as e:
            raise RuntimeError(f"Failed to expand word boundaries for pattern '{pattern}': {str(e)}")


    # find wildcard matches
    @staticmethod
    def find_matches(patterns, decoded_text: str, char_to_byte_map):
        """
        Find all matches. These are also passed on to fuzzy match.
        """	
        try:
            matches = []
            
            # Pre-compile all patterns once - cached version
            compiled_patterns = {}
            for pattern in patterns:
                if not pattern:
                    continue
                if '*' in pattern or '?' in pattern:
                    regex_pattern = SnippetExtractor.wildcards_to_regex(pattern)
                    compiled_patterns[pattern] = re.compile(regex_pattern, re.IGNORECASE | re.DOTALL)
                else:
                    escaped_pattern = re.escape(pattern)
                    regex_pattern = r'\b' + escaped_pattern + r'\b'
                    compiled_patterns[pattern] = re.compile(regex_pattern, re.IGNORECASE)

            for pattern, compiled_pattern in compiled_patterns.items():
                try:
                    # Check stop event before each iteration
                    for match in compiled_pattern.finditer(decoded_text):
                        start_pos, end_pos = match.start(), match.end()
                        match_text = decoded_text[start_pos:end_pos]

                        if '*' in pattern or '?' in pattern:
                            expanded_match_text, expanded_start, expanded_end = SnippetExtractor.expand_to_word_boundaries(
                                decoded_text, start_pos, end_pos, pattern
                            )
                            match_text = expanded_match_text
                            start_pos = expanded_start
                            end_pos = expanded_end

                        match_obj = Match(pattern, match_text, start_pos, end_pos)
                        match_obj.set_byte_positions(char_to_byte_map)
                        matches.append(match_obj)
                except re.error as e:
                    raise RuntimeError(f"Regex compilation error for pattern '{pattern}': {str(e)}")

            return matches
        except Exception as e:
            raise RuntimeError(f"Failed to find matches: {str(e)}")


    # distance check of all found matches
    @staticmethod
    def filter_by_distance(matches, distance: int, buzzwords):
        """
        filter matches by distance limit given by user input.
        """
        try:
            if not matches:
                return []
            
            # Use sets for faster membership checks and avoid redundant lookups
            pattern_positions = {word: set() for word in buzzwords}
            for m in matches:
                if m.pattern in pattern_positions:
                    pattern_positions[m.pattern].add((m.start_char, m.end_char))
                    
            if any(not pos_set for pos_set in pattern_positions.values()):
                return []
                
            combined_spans = []
            first_word = list(buzzwords)[0]
            
            for start1, end1 in pattern_positions[first_word]:
                span_candidates = [(start1, end1)]
                for other_word in buzzwords:
                    if other_word == first_word:
                        continue
                    best_match = None
                    min_distance = float('inf')
                    
                    # Direct set iteration - much faster than list lookup
                    for start2, end2 in pattern_positions[other_word]:
                        dist = abs(start1 - start2)
                        if dist <= distance and dist < min_distance:
                            min_distance = dist
                            best_match = (start2, end2)
                            
                    if best_match:
                        span_candidates.append(best_match)
                        
                if len(span_candidates) == len(buzzwords):
                    min_pos = min(s for s, _ in span_candidates)
                    max_pos = max(e for _, e in span_candidates)
                    combined_spans.append((min_pos, max_pos))
                    
            return combined_spans
        except Exception as e:
            raise RuntimeError(f"Failed to filter by distance: {str(e)}")

    # snippet extraction, pre_ratio and post_ratio given from user
    @staticmethod
    def extract_snippets(matches, snippet_size, pre_ratio, post_ratio, decoded_text):
        try:
            snippets = []
            for start, end in matches:
                pre_chars = int(snippet_size * pre_ratio)
                post_chars = int(snippet_size * post_ratio)
                snippet_start = max(0, start - pre_chars)
                snippet_end = min(len(decoded_text), end + post_chars)
                snippets.append((snippet_start, snippet_end))
            return snippets
        except Exception as e:
            raise RuntimeError(f"Failed to extract snippets: {str(e)}")
    
    # merge snippet if overlapping
    @staticmethod
    def merge_snippets(snippets):
        try:
            if not snippets:
                return [], 0
                
            total_snippets = len(snippets)
            
            # Sort once instead of repeatedly during merging
            sorted_snippets = sorted(snippets, key=lambda x: x[0])
            merged = [sorted_snippets[0]]
            
            for current in sorted_snippets[1:]:
                last_end = merged[-1][1]
                if current[0] <= last_end:
                    # Fast merge - no need to check all previous ones
                    merged[-1] = (merged[-1][0], max(last_end, current[1]))
                else:
                    merged.append(current)
                    
            return merged, total_snippets
        except Exception as e:
            raise RuntimeError(f"Failed to merge snippets: {str(e)}")


    # ----------
    # Fuzzy part
    # ----------
    # use results of wildcard find_matches for fuzzy search
    @staticmethod
    def find_fuzzy_matches(decoded_text: str, wildcard_matches: List[Match], threshold: float, stop_event=None):
        """
        Search the entire text using matches from wildcard search as fuzzily searched words.
        Returns list of tuples (match_start, match_end, score, original_word) where score >= threshold.
        """
        try:
            fuzzy_results = []
            
            # Get all unique texts from wildcard matches to use as buzzwords
            buzzwords = [match.text for match in wildcard_matches if match.text.strip()]
            
            if not buzzwords:
                return fuzzy_results
                
            # Use rapidfuzz.process.extract for efficient fuzzy matching
            # Process each word in the text against our buzzwords
            words = decoded_text.split()
            processed_words = []
            
            # Create a list of (word, start_pos, end_pos) tuples to track positions
            current_pos = 0
            for word in words:
                if stop_event and stop_event.is_set():
                    raise RuntimeError("Fuzzy search was aborted")
                    
                # Find exact position of this word in original text
                try:
                    pos = decoded_text.index(word, current_pos)
                    processed_words.append((word, pos, pos + len(word)))
                    current_pos = pos + len(word)
                except ValueError:
                    # Word not found - skip it
                    continue
            
            # For each word in the document, check fuzzy matches against our buzzwords
            for word, start_pos, end_pos in processed_words:
                if stop_event and stop_event.is_set():
                    raise RuntimeError("Fuzzy search was aborted")
                    
                # Find best match among buzzwords using rapidfuzz
                try:
                    # Get top match with score >= threshold
                    matches = process.extract(
                        word, 
                        buzzwords, 
                        limit=1,
                        scorer=fuzz.ratio,
                        score_cutoff=threshold
                    )
                    
                    if matches and len(matches) > 0:
                        best_match_text, score, _ = matches[0]
                        # Add the position of this match in original text + the actual word that was matched
                        fuzzy_results.append((start_pos, end_pos, score, word))
                        
                except Exception as e:
                    # Continue with other words if one fails
                    continue
                    
            return fuzzy_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to find fuzzy matches: {str(e)}")



    # filter by distance for fuzzy matches if "AND
    # filter by distance for fuzzy matches - NEW IMPLEMENTATION
    @staticmethod
    def filter_by_distance_fuzzy(fuzzy_matches, distance_threshold):
        """
        Filter fuzzy matches requiring all buzzwords within distance threshold.
        Groups matching words together and only keeps groups where all required 
        buzzwords appear within the specified distance.
        
        Args:
            fuzzy_matches: List of tuples (start_pos, end_pos, score, original_word)
            distance_threshold: Maximum character distance between matches
            
        Returns:
            List of filtered fuzzy match tuples
        """
        try:
            if not fuzzy_matches:
                return []
                
            # Group matches by their original word (buzzword)
            word_groups = {}
            for start, end, score, word in fuzzy_matches:
                if word not in word_groups:
                    word_groups[word] = []
                word_groups[word].append((start, end, score))
            
            # Debugging output
            print(f"DEBUG: Processing {len(word_groups)} unique words from fuzzy matches")
            for word, positions in word_groups.items():
                print(f"  Word '{word}': {len(positions)} matches at positions {[pos[0] for pos in positions]}")
            
            # Get all buzzwords that were actually found
            found_buzzwords = list(word_groups.keys())
            
            if len(found_buzzwords) < 2:
                print("DEBUG: Only one unique word found - returning all matches")
                return fuzzy_matches
            
            # For multiple words, create sliding windows to find valid groups
            # This approach checks each possible combination of positions for different words
            results = []
            
            # Sort all positions by start position to make grouping easier
            all_positions = []
            for word, pos_list in word_groups.items():
                for start, end, score in pos_list:
                    all_positions.append((start, end, score, word))
            
            all_positions.sort(key=lambda x: x[0])  # Sort by start position
            
            print(f"DEBUG: Total positions to process: {len(all_positions)}")
            
            # Try to find groups where multiple buzzwords appear within distance
            i = 0
            while i < len(all_positions):
                current_start = all_positions[i][0]
                current_end = all_positions[i][1]
                
                # Create a window around this position
                window_end = current_start + distance_threshold
                
                # Collect all words in this window
                window_words = {}
                j = i
                while j < len(all_positions) and all_positions[j][0] <= window_end:
                    pos_start, pos_end, score, word = all_positions[j]
                    if word not in window_words:
                        window_words[word] = []
                    window_words[word].append((pos_start, pos_end, score))
                    j += 1
                
                # Check if we have matches for ALL required buzzwords
                if len(window_words) >= 2:  # At least two different words found together
                    # For now, just return all the original matches from this window
                    # This is a simpler approach - you could get more sophisticated later
                    print(f"DEBUG: Found group with {len(window_words)} words in range [{current_start}, {window_end}]")
                    for word, positions in window_words.items():
                        print(f"  Word '{word}': {[pos[0] for pos in positions]}")
                    
                    # Add all matches from this valid window
                    for word, positions in window_words.items():
                        for start, end, score in positions:
                            results.append((start, end, score, word))
                else:
                    print(f"DEBUG: Window [{current_start}, {window_end}] only had {len(window_words)} unique words")
                
                i = j
            
            # Remove duplicates while preserving order
            seen = set()
            final_results = []
            for item in results:
                if item not in seen:
                    seen.add(item)
                    final_results.append(item)
            
            print(f"DEBUG: Final filtered results count: {len(final_results)}")
            return final_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to filter fuzzy matches by distance: {str(e)}")





    # extract fuzzy snippets
    @staticmethod
    def extract_snippets_fuzzy(matches, snippet_size, pre_ratio, post_ratio, decoded_text):
        """
        Extract snippets from fuzzy matches.
        """
        try:
            snippets = []
            for start, end, score, original_word in matches:
                # Apply ratio-based padding to include more context
                pre_chars = int(snippet_size * pre_ratio)
                post_chars = int(snippet_size * post_ratio)
                snippet_start = max(0, start - pre_chars)
                snippet_end = min(len(decoded_text), end + post_chars)
                
                snippets.append((snippet_start, snippet_end, score, original_word))
            return snippets
        except Exception as e:
            raise RuntimeError(f"Failed to extract fuzzy snippets: {str(e)}")

    

    # merge fuzzy snippets
    @staticmethod
    def merge_snippets_fuzzy(snippets):
        """
        Merge overlapping or adjacent fuzzy snippets.
        """
        try:
            if not snippets:
                return [], 0
                
            total_snippets = len(snippets)
            
            # Sort by start position
            sorted_snippets = sorted(snippets, key=lambda x: x[0])
            merged = [sorted_snippets[0]]
            
            for current in sorted_snippets[1:]:
                last_end = merged[-1][1]
                
                if current[0] <= last_end:
                    # Merge overlapping or adjacent snippets
                    new_start = merged[-1][0]
                    new_end = max(last_end, current[1])
                    
                    # Update the score to be average of both scores (or keep highest)
                    avg_score = (merged[-1][2] + current[2]) / 2.0
                    
                    merged[-1] = (new_start, new_end, avg_score, merged[-1][3])  # Keep original word from first
                else:
                    merged.append(current)
                    
            return merged, total_snippets
        except Exception as e:
            raise RuntimeError(f"Failed to merge fuzzy snippets: {str(e)}")




# --- Main search function ---

def run_search_for_file(file_path: str, config: dict, stop_event: threading.Event):
    """
    Run search for a single file. Writes output_snippets.txt and output_fuzzy_snippets.txt.
    Returns (wildcard_text, fuzzy_text) strings for UI display.
    Optimized version with faster operations.
    """
    try:
        processor = TextProcessor(file_path)
        buzzwords = [bw for bw in config.get("buzzwords", []) if bw.strip()]

        # Use set for filter_by_distance membership but keep list for order preservation
        buzzwords_set = list(dict.fromkeys(buzzwords))  # unique preserving order

        # wildcard-part - optimized
        all_matches = SnippetExtractor.find_matches(
            buzzwords_set,
            processor.decoded_text,
            processor.char_to_byte
        )

        if config.get("search_type", "AND") == "AND":
            final_matches = SnippetExtractor.filter_by_distance(
                all_matches,
                config.get("distance_match", 100),
                buzzwords_set
            )
        else:
            final_matches = [(m.start_char, m.end_char) for m in all_matches]

        snippets = SnippetExtractor.extract_snippets(
            final_matches,
            config.get("snippet_size", 2000),
            config.get("pre_ratio", 0.3),
            config.get("post_ratio", 0.7),
            processor.decoded_text
        )

        merged_snippets, total_snippets = SnippetExtractor.merge_snippets(snippets)

        # Build wildcard textual output - optimized with pre-calculated values
        wildcard_blocks = []
        for idx, (start, end) in enumerate(merged_snippets):
            if stop_event.is_set():
                raise RuntimeError("Search was aborted")

            s_b = processor.char_to_byte[start]
            e_b = processor.char_to_byte[end]
            snippet_bytes = processor.text_bytes[s_b:e_b]
            snippet_text = snippet_bytes.decode("utf-8", errors="surrogateescape")
            cleaned = re.sub(r'\s+', ' ', snippet_text) # without \n and \r

            # Find first match
            match_text = None
            byte_start = None
            for m in all_matches:
                if start <= m.start_char and end >= m.end_char:
                    match_text = m.text
                    byte_start = m.byte_start
                    break

            block = [
                {"Excerpt": idx + 1},
                {"Match Text": match_text},
                {"Start position, match_text": byte_start},
                {"Content": cleaned},
            ]
            wildcard_blocks.append(json.dumps(block, ensure_ascii=False, indent=1))

        wildcard_text = "\n\n".join(wildcard_blocks)

        # fuzzy part, similar approach like wildcard
        ft = config.get("fuzzy_threshold", 96)
        if not isinstance(ft, (int, float)) or not (0 <= ft <= 100):
            ft = 96.0  # default threshold

        # Use all wildcard matches as input for fuzzy search
        fuzzy_matches = SnippetExtractor.find_fuzzy_matches(
            processor.decoded_text,
            all_matches,
            ft
        )

        if config.get("search_type", "AND") == "AND":
            if len(buzzwords) > 1:
                filtered_fuzzy_matches = SnippetExtractor.filter_by_distance_fuzzy(
                    fuzzy_matches,
                    config.get("distance_match", 100)
                )
            else:
                # fallback to OR behavior when only one buzzword
                filtered_fuzzy_matches = fuzzy_matches
        else:
            filtered_fuzzy_matches = fuzzy_matches

        # Extract snippets for fuzzy matches
        fuzzy_snippets = SnippetExtractor.extract_snippets_fuzzy(
            filtered_fuzzy_matches,
            config.get("snippet_size", 2000),
            config.get("pre_ratio", 0.3),
            config.get("post_ratio", 0.7),
            processor.decoded_text
        )

        # Merge fuzzy snippets
        merged_fuzzy_snippets, total_fuzzy_snippets = SnippetExtractor.merge_snippets_fuzzy(fuzzy_snippets)

        # Build fuzzy textual output - now with actual matched text and byte positions
        fuzzy_blocks = []
        for idx, (start, end, score, original_word) in enumerate(merged_fuzzy_snippets):
            if stop_event.is_set():
                raise RuntimeError("Search was aborted")

            s_b = processor.char_to_byte[start]
            e_b = processor.char_to_byte[end]
            snippet_bytes = processor.text_bytes[s_b:e_b]
            snippet_text = snippet_bytes.decode("utf-8", errors="surrogateescape")
            cleaned_snippet = re.sub(r'\s+', ' ', snippet_text) # without \n and \r

            # Get the actual byte start position of the matched word in the original file
            match_byte_start = None
            for fm in fuzzy_matches:  # Use original fuzzy_matches, not filtered_fuzzy_matches
                if fm[3] == original_word and fm[0] >= start and fm[1] <= end:
                    # Found the exact fuzzy match that corresponds to this merged snippet
                    match_byte_start = processor.char_to_byte[fm[0]]
                    break

            block = [
                {"Excerpt": idx + 1},
                {"Match Text": original_word},  # Show the actual word that was matched
                {"Score": score},
                {"Start Byte Position": match_byte_start},  # Add byte position to JSON output
                {"Content": cleaned_snippet},
            ]
            fuzzy_blocks.append(json.dumps(block, ensure_ascii=False, indent=1))


        fuzzy_text = "\n\n".join(fuzzy_blocks)

        return wildcard_text, fuzzy_text
    except Exception as e:
        raise RuntimeError(f"Search failed for file {file_path}: {str(e)}")

# ---
# GUI
# ---

class SearchThread(threading.Thread):
    def __init__(self, paths, config, stop_event, on_complete):
        super().__init__()
        self.paths = paths
        self.config = config
        self.stop_event = stop_event
        self.on_complete = on_complete  # callback(wildcard_text, fuzzy_text, finished_ok)

    def run(self):
        try:
            agg_wild = []
            agg_fuzzy = []
            for p in self.paths:
                if self.stop_event.is_set():
                    self.on_complete("", "", False)
                    return
                try:
                    w, f = run_search_for_file(p, self.config, self.stop_event)
                    agg_wild.append(w)
                    agg_fuzzy.append(f)
                except Exception as e:
                    # If one file fails, continue with others but report the error
                    if not self.stop_event.is_set():  # Only show error if not aborted
                        self.on_complete(f"ERROR processing {p}: {str(e)}", f"ERROR processing {p}: {str(e)}", False)
                        return
            wildcard_text = "\n\n--- FILE BOUNDARY ---\n\n".join(agg_wild)
            fuzzy_text = "\n\n--- FILE BOUNDARY ---\n\n".join(agg_fuzzy)
            self.on_complete(wildcard_text, fuzzy_text, True)
        except Exception as e:
            # Handle exceptions in the thread itself
            self.on_complete(f"THREAD ERROR: {str(e)}", f"THREAD ERROR: {str(e)}", False)

class MainFrame(wx.Frame):
    def __init__(self):
        super().__init__(None, title="Text Search by Sevenof9 (v3_alpha)", size=(1200, 1000))
        panel = wx.Panel(self)

        # Top: file / dir pickers and right-side label for chosen path
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.file_picker = wx.FilePickerCtrl(panel, style=wx.FLP_OPEN | wx.FLP_FILE_MUST_EXIST)
        self.dir_picker = wx.DirPickerCtrl(panel)
        self.path_label = wx.StaticText(panel, label="No file/folder selected")

        top_sizer.Add(self.file_picker, 0, wx.ALL | wx.ALIGN_LEFT, 4)
        top_sizer.Add(self.dir_picker, 0, wx.ALL | wx.ALIGN_LEFT, 4)
        top_sizer.Add(self.path_label, 0, wx.ALL | wx.ALIGN_LEFT, 6)

        # Middle: left = buzzwords (4 fields with AND/OR buttons between), right = controls/config
        middle_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left: buzzwords area
        buzz_sizer = wx.BoxSizer(wx.VERTICAL)
        self.buzz_inputs = []
        self.toggle_buttons = []
        for i in range(4):
            txt = wx.TextCtrl(panel, size=(250, -1))
            self.buzz_inputs.append(txt)
            buzz_sizer.Add(txt, 0, wx.ALL | wx.ALIGN_LEFT, 2)
            if i < 3:
                btn = wx.Button(panel, label="AND", size=(80, 24))
                btn.Bind(wx.EVT_BUTTON, self.on_toggle)
                self.toggle_buttons.append(btn)
                buzz_sizer.Add(btn, 0, wx.ALL | wx.ALIGN_LEFT, 2)

        middle_sizer.Add(buzz_sizer, 0, wx.ALL | wx.ALIGN_LEFT, 6)

        # Right: controls and config
        ctrl_sizer = wx.BoxSizer(wx.VERTICAL)

        # Start / Abort
        self.start_button = wx.Button(panel, label="Start Search")
        self.abort_button = wx.Button(panel, label="Abort")
        self.abort_button.Disable()
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        self.abort_button.Bind(wx.EVT_BUTTON, self.on_abort)
        ctrl_sizer.Add(self.start_button, 0, wx.ALL | wx.ALIGN_LEFT, 4)
        ctrl_sizer.Add(self.abort_button, 0, wx.ALL | wx.ALIGN_LEFT, 4)

        # Config fields
        self.cfg_fields = {}
        defaults = [("snippet_size", "2000"),
                    ("pre_ratio", "0.3"),
                    ("post_ratio", "0.7"),
                    ("distance_match", "300"),
                    ("fuzzy_threshold", "96")]
        for label, val in defaults:
            row = wx.BoxSizer(wx.HORIZONTAL)
            lbl = wx.StaticText(panel, label=label + ":")
            fld = wx.TextCtrl(panel, value=val, size=(50, -1))
            # Bind focus event for validation
            fld.Bind(wx.EVT_KILL_FOCUS, self.on_field_focus_lost)
            row.Add(lbl, 0, wx.ALL | wx.ALIGN_LEFT, 2)
            row.Add(fld, 0, wx.ALL | wx.ALIGN_LEFT, 2)
            ctrl_sizer.Add(row, 0, wx.ALL | wx.ALIGN_LEFT, 2)
            self.cfg_fields[label] = fld

        middle_sizer.Add(ctrl_sizer, 0, wx.ALL | wx.ALIGN_LEFT, 6)

        # Bottom: results (wildcard and fuzzy) across full width
        result_sizer = wx.BoxSizer(wx.VERTICAL)
        result_sizer.Add(wx.StaticText(panel, label="Wildcard Results (output_snippets.txt):"), 0, wx.ALL | wx.ALIGN_LEFT, 2)
        self.wildcard_box = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 220))
        result_sizer.Add(self.wildcard_box, 1, wx.EXPAND | wx.ALL, 4)
        result_sizer.Add(wx.StaticText(panel, label="Fuzzy Results (output_fuzzy_snippets.txt):"), 0, wx.ALL | wx.ALIGN_LEFT, 2)
        self.fuzzy_box = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(-1, 220))
        result_sizer.Add(self.fuzzy_box, 1, wx.EXPAND | wx.ALL, 4)

        # Main vertical layout using only horizontal alignment flags where appropriate
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(top_sizer, 0, wx.ALL | wx.ALIGN_LEFT, 6)
        main_sizer.Add(middle_sizer, 0, wx.ALL | wx.ALIGN_LEFT, 6)
        main_sizer.Add(result_sizer, 1, wx.EXPAND | wx.ALL, 6)

        panel.SetSizer(main_sizer)

        # Events
        self.file_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self.on_path_change)
        self.dir_picker.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_path_change)

        # Thread controls
        self.worker = None
        self.stop_event = threading.Event()

    def on_field_focus_lost(self, evt):
        """Validate all fields when any field loses focus"""
        self.validate_all_fields()
        evt.Skip()  # Allow normal processing to continue

    def validate_all_fields(self):
        """Validate all configuration fields and enforce dependencies"""
        try:
            # Get current values
            snippet_size_val = self.cfg_fields["snippet_size"].GetValue().strip()
            pre_ratio_val = self.cfg_fields["pre_ratio"].GetValue().strip()
            post_ratio_val = self.cfg_fields["post_ratio"].GetValue().strip()
            distance_match_val = self.cfg_fields["distance_match"].GetValue().strip()
            fuzzy_threshold_val = self.cfg_fields["fuzzy_threshold"].GetValue().strip()

            # Default values if empty
            snippet_size_val = snippet_size_val if snippet_size_val else "2000"
            pre_ratio_val = pre_ratio_val if pre_ratio_val else "0.3"
            post_ratio_val = post_ratio_val if post_ratio_val else "0.7"
            distance_match_val = distance_match_val if distance_match_val else "300"
            fuzzy_threshold_val = fuzzy_threshold_val if fuzzy_threshold_val else "96"

            # Validate and process each field
            # snippet_size: min=0, max=999999, round to integer
            snippet_size = int(float(snippet_size_val)) if snippet_size_val else 2000
            snippet_size = max(0, min(999999, snippet_size))

            # pre_ratio: min=0.1, max=0.9, 1 decimal place
            pre_ratio = round(float(pre_ratio_val), 1) if pre_ratio_val else 0.3
            pre_ratio = max(0.1, min(0.9, pre_ratio))

            # post_ratio: min=0.1, max=0.9, 1 decimal place
            post_ratio = round(float(post_ratio_val), 1) if post_ratio_val else 0.7
            post_ratio = max(0.1, min(0.9, post_ratio))

            # Ensure pre + post = 1 (adjust one to maintain sum)
            total = pre_ratio + post_ratio
            if abs(total - 1.0) > 0.001:  # Allow small floating point differences
                # Adjust post_ratio to make the sum equal to 1.0
                post_ratio = round(1.0 - pre_ratio, 1)
                self.cfg_fields["post_ratio"].SetValue(str(post_ratio))

            # distance_match: min=0, max=snippet_size, round to integer
            distance_match = int(float(distance_match_val)) if distance_match_val else 300
            distance_match = max(0, min(snippet_size, distance_match))

            # If snippet_size < distance_match, adjust snippet_size to match
            if snippet_size < distance_match:
                snippet_size = distance_match
                self.cfg_fields["snippet_size"].SetValue(str(snippet_size))

            # fuzzy_threshold: min=1, max=100, round to integer
            fuzzy_threshold = int(float(fuzzy_threshold_val)) if fuzzy_threshold_val else 96
            fuzzy_threshold = max(1, min(100, fuzzy_threshold))

            # Apply validated values back to fields
            self.cfg_fields["snippet_size"].SetValue(str(snippet_size))
            self.cfg_fields["pre_ratio"].SetValue(str(pre_ratio))
            self.cfg_fields["post_ratio"].SetValue(str(post_ratio))
            self.cfg_fields["distance_match"].SetValue(str(distance_match))
            self.cfg_fields["fuzzy_threshold"].SetValue(str(fuzzy_threshold))

        except Exception as e:
            # If validation fails, show error but don't block the user
            wx.MessageBox(f"Validation Error: {str(e)}", "Error")

    def on_path_change(self, evt):
        path = evt.GetPath()
        self.path_label.SetLabel(path)

    def on_toggle(self, evt):
        btn = evt.GetEventObject()
        label = btn.GetLabel()
        if label == "AND":
            btn.SetLabel("OR")
        else:
            btn.SetLabel("AND")

    def on_abort(self, evt):
        """Abort button now properly stops all processes"""
        if self.worker and self.worker.is_alive():
            # Set the stop event to signal all running operations to abort
            self.stop_event.set()

            # Disable buttons immediately
            self.abort_button.Disable()
            self.start_button.Enable()

            # Clear any text that might have been set during processing
            wx.CallAfter(self.wildcard_box.SetValue, "Aborting...")
            wx.CallAfter(self.fuzzy_box.SetValue, "Aborting...")

    def on_start(self, evt):
        # get path
        path = self.path_label.GetLabel()
        if not path or path == "No file/folder selected":
            wx.MessageBox("Please select a file or folder first.", "Error")
            return

        try:
            if os.path.isdir(path):
                txts = [str(Path(path) / f) for f in sorted(os.listdir(path))
                        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(path, f))]
                if not txts:
                    wx.MessageBox("Selected folder contains no .txt files.", "Error")
                    return
                paths = txts
            else:
                if not os.path.isfile(path):
                    wx.MessageBox("Selected path is not a file.", "Error")
                    return
                # Only allow .txt files - this validation was missing before
                if not path.lower().endswith(".txt"):
                    wx.MessageBox("Please select a .txt file.", "Error")
                    return
                paths = [path]
        except Exception as e:
            wx.MessageBox(f"Failed to access path: {str(e)}", "Error")
            return

        # prepare config
        try:
            cfg = {
                "snippet_size": int(self.cfg_fields["snippet_size"].GetValue().strip()),
                "pre_ratio": float(self.cfg_fields["pre_ratio"].GetValue().strip()),
                "post_ratio": float(self.cfg_fields["post_ratio"].GetValue().strip()),
                "distance_match": int(self.cfg_fields["distance_match"].GetValue().strip()),
                "fuzzy_threshold": float(self.cfg_fields["fuzzy_threshold"].GetValue().strip()),
            }
        except Exception:
            wx.MessageBox("Please check numeric configuration values.", "Error")
            return

        buzzwords = [t.GetValue().strip() for t in self.buzz_inputs]
        cfg["buzzwords"] = buzzwords
        cfg["search_type"] = "AND" if self.toggle_buttons[0].GetLabel() == "AND" else "OR"

        # UI state
        self.start_button.Disable()
        self.abort_button.Enable()
        self.wildcard_box.SetValue("Running...")
        self.fuzzy_box.SetValue("Running...")

        # reset stop_event and start thread
        self.stop_event.clear()

        # Overwrite output files at the beginning of each new search
        Path("output_snippets.txt").write_text("", encoding="utf-8", errors="surrogateescape")
        Path("output_fuzzy_snippets.txt").write_text("", encoding="utf-8", errors="surrogateescape")

        self.worker = SearchThread(paths, cfg, self.stop_event, self.on_search_complete)
        self.worker.start()

    def on_search_complete(self, wildcard_text, fuzzy_text, finished_ok):
        # This callback runs in worker thread; must marshal to main GUI thread
        def _update():
            if finished_ok:
                self.wildcard_box.SetValue(wildcard_text)
                self.fuzzy_box.SetValue(fuzzy_text)

                # Append results to output files for each processed file
                Path("output_snippets.txt").write_text(wildcard_text, encoding="utf-8", errors="surrogateescape")
                Path("output_fuzzy_snippets.txt").write_text(fuzzy_text, encoding="utf-8", errors="surrogateescape")

            else:
                # signals either error or aborted
                self.wildcard_box.SetValue(wildcard_text or "Aborted / Error")
                self.fuzzy_box.SetValue(fuzzy_text or "Aborted / Error")
            self.stop_event.clear()
            self.start_button.Enable()
            self.abort_button.Disable()

        wx.CallAfter(_update)

if __name__ == "__main__":
    app = wx.App(False)
    frame = MainFrame()
    frame.Show()
    app.MainLoop()




