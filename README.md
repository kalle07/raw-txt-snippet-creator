# Raw TXT Snippet Creator

Search large collections of plain-text books and documents from a local desktop application.<br>
v04 new, much faster indexing, much faster searching, need new index!<br>

**Download:** [TextSearch-by-kalle07.exe (~200 MB)](https://github.com/kalle07/raw-txt-snippet-creator/releases/download/_v04/TextSearch-by-kalle07.exe)

📥 Downloads: <!--download-count-->031<!--/download-count-->

<img width="2239" height="1259" alt="Raw TXT Snippet Creator screenshot" src="https://github.com/user-attachments/assets/1c323d1e-4588-40e4-a500-ab9fdfb081c5" />

## Features

- Search up to **three keywords** together (AND).
- Find passages where keywords occur within a configurable character distance.
- Extract and merge overlapping snippets.
- Exact, wildcard, regex and fuzzy search.
- Case-sensitive search.
- Separate **Literal text search** mode.
- Folder-level word counts and search-result counts.
- Browse results, inspect matching passages and save snippets.
- Automatically monitor TXT folders for file changes.

Everything runs locally. No embedding model, cloud service or internet connection is required during searching.

## Two independent search modes

### 1. Database search (default)

Uses the searchable index for fast vocabulary-based searches.

- All entered keywords must be found in the same document.
- Keywords must satisfy the configured maximum character distance.
- Partial words require wildcards.
- Supports exact, wildcard, regex and fuzzy matching.
- Case sensitivity can be configured individually.

**Examples**

| Search | Text in document | Result |
|---|---|---|
| `zins` | `Zinssatz` | No match |
| `zins*` | `Zinssatz` | Match |
| `hello wor` | `hello world` | No match |
| `hello wor*` | `hello world` | Match |
| `*llo world` | `hello world` | Match |
| `??llo world` | `hello world` | Match |
| `?llo world` | `hello world` | No Match |
| `* world` | `hello world` and `one world` and `...` | Match |



Wildcards:
- `?` matches one character.
- `*` matches up to 100 visible characters without crossing whitespace or a line break.

Regex and fuzzy search are available through their respective keyword settings.

An unresolved search term does not automatically trigger a full-text search.

### 2. Literal text search (optional checkbox)

Bypasses vocabulary and postings and searches the original physical TXT files directly.

It searches character sequences, partial words and complete phrases.

**Examples**

| Search | Text in document | Result |
|---|---|---|
| `hello wor` | `hello world` | Match |
| `llo world` | `hello world` | Match |
| `ell worl` | `hello world` | Match |
| `zins gesetz` | `Basiszinssatz und Gesetz` | Match |

Important:
- `*` and `?` are ordinary characters, not wildcards.
- Regex and fuzzy matching are not used.
- Case-insensitive matching supports English and German letters, including Ä/ä, Ö/ö, Ü/ü and ẞ/ß.
- `straße` does not match `STRASSE`.
- New files must be indexed before they are included in searches.

Literal search uses physical CPU cores and parallel batches of 100 files.

## Understanding search results

A result represents a document containing all required keywords within the configured distance.

The output distinguishes:

- **Individual term occurrences:** How often each keyword occurs in the document.
- **Proximity matches:** How many matching passages were found.
- **Snippets:** Extracted text surrounding matching passages.

A high occurrence count does not necessarily mean that every occurrence belongs to a valid proximity match. The distance condition determines which passages qualify.

Use the document browser to inspect the complete context, navigate between matches and save useful snippets through the context menu.

## Search settings

All distances are measured in **characters**, not words.

- `Max_Distance_Chars`: Maximum allowed distance between search terms.
- `Snippet_Context_Chars`: Amount of text shown before and after a match.
- `Max documents`: Maximum number of returned documents.
- `Max results per document`: Limits matching passages per document.

## Getting started

### Windows EXE

1. Download and run the EXE.
2. On first launch, the application creates `text_search.lancedb` and the `txt` folder.
3. Copy your `.txt` files into `txt`. Subfolders are supported.
4. Click **Index new files**.
5. Enter your keywords and search.

The `deleted` folder is located next to `txt`, not inside it.

### Run from Python

Python 3.11 or newer is recommended.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python start.py
```

Place `start.py`, `gui.py` and `engine.py` in the project folder.

## Performance

Designed for large local TXT collections.

Example from testing:
- Approximately 6,000 TXT files (~20E+06 words).
- Approximately 250 MB of text.
- Initial indexing: approximately one minute.

Search performance in this case depends on the selected mode, query, file sizes, CPU and storage speed. Database search is generally faster (~100ms); literal search scans the original files and takes longer (max 2000ms).

## Database compatibility

Index formats may change between versions. When upgrading to an incompatible version, close the application and delete `text_search.lancedb`. The database will be recreated and the TXT collection must be indexed again.

Keep a backup if you need the previous index.

## Related project

For converting PDFs into searchable TXT files, see [PDF Parser](https://github.com/kalle07/pdf2txt-parser).

## Notes

- Windows is the only platform tested.
- Only plain `.txt` files are searched. Convert PDFs first.
- All searching is performed locally.

**Disclaimer:** Use this software at your own responsibility. I am not responsible for errors, data loss or system problems.
