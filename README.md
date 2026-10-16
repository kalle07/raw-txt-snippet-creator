# raw-txt-snippet-creator with database


https://github.com/kalle07/raw-txt-snippet-creator/releases/download/_v02/TextSearch-by-kalle07.exe<br>
<br>
📥 Downloads: <!--download-count-->016<!--/download-count-->
<br>
<br>

Search large collections of plain-text books and documents from a
desktop app.

no install selfrunnuing APP - ~200MB
<img width="2239" height="1259" alt="grafik" src="https://github.com/user-attachments/assets/1c323d1e-4588-40e4-a500-ab9fdfb081c5" />


Instead of opening files one by one, enter up to **three keywords** and
find passages where they occur close together. The app extracts readable
snippets around matches, so you can quickly review relevant passages and
copy them into your notes or chat.

It is a lightweight, local text-search approach: it works with ordinary
`.txt` files and a searchable index rather than requiring an embedding
model.

## What it does

-   **Search up to three keywords together.** Keywords are linked with
    **AND**: a result must satisfy the search terms within the
    configured character distance.
-   **Find passages, not just hit counts.** The snippet extractor
    returns context around matching terms.
-   **Merge overlapping snippets.** When extracted sections overlap,
    they are merged instead of duplicated. The combined output does not
    exceed the source text length.
-   **Choose your search method for each keyword:**
    -   **Exact / regex / wildcard search** for precise terms and
        patterns.
    -   **Fuzzy search** to find small spelling variations.
    -   **Case-sensitive search** when capitalization matters.
-   **Phrase search** is supported.
-   **Browse results in the document browser** and use the right-click
    menu to save snippets.
-   **Monitor your TXT collection while the app is open.** File changes
    update the GUI tree; use **Index new files** to update the search
    database.

## Search tips

### Distance and context are measured in characters

-   **`Max_Distance_Chars`** sets the maximum distance between the
    keywords, in characters.
-   **`Snippet_Context_Chars`** sets how many characters of context to
    include before and after a match.

All distances and context values are measured in **characters**, not
words or tokens. As a rough guide, 4,000 characters are approximately
1,000 tokens---around one page of a book, depending on formatting and
language.

### Exact, wildcard and fuzzy search

In exact/wildcard mode, a partial word is not automatically treated as a
prefix. For example, searching for `friend` does not find `friendship`;
use `friend*` to match a word beginning with `friend`.

-   `?` matches one character.
-   `*` matches up to 100 visible characters without crossing whitespace
    or a line break.
-   Regex patterns can also be used when regex search is enabled.

Fuzzy search is useful for typos and minor spelling differences. A fuzzy
distance of **1** generally allows one character difference; **2** can
return many more variations, so use it when broader matching is desired.

## Getting started

### Using the EXE

1.  Copy the EXE file into a folder of your choice.
2.  Run it. On first launch, the app creates the local database
    (`text_search.lancedb`) and the `txt` folder if they do not already
    exist.
3.  Copy `.txt` files into `txt`, or choose text files through the app.
    Subfolders inside `txt` are supported.
4.  Click **Index new files** to add new files or update changed files.
5.  Enter one, two or three keywords, configure the distance and
    context, and search.

The `deleted` folder is located next to `txt`, not inside it. The app
searches `.txt` files throughout `txt` and its subfolders.

### Running from Python

If you are running the source code instead of the packaged EXE, place
`gui.py` and `engine.py` in the same folder, then install the
dependencies:

``` bash
<folder of your choice>
python -m venv venv
venv\Scripts\activate # On Windows
pip install -r requirements.txt
python start.py
```

Python 3.11 or newer is recommended. wxPython installation on Linux may
require system GTK/wxWidgets libraries.

## Performance and collection size

The app is designed for large local TXT collections. In testing, it has
been used with large book-sized text files and collections of many
files. Actual indexing and search times depend on file sizes, storage
speed, CPU and search settings.

As an example from testing, indexing ten large books took approximately
**5--10 seconds**, loading an existing database took **under one
second**, and a typical search took **under 100 ms**. These are
indicative results, not guarantees for every computer or collection.

The first index takes longer because the app must read and process the
source files. Later launches reuse the local database.

## Important: database compatibility

A new app version may change the index format. **There is no guaranteed
backward compatibility between versions.** When upgrading to a version
that changes the index, close the app and delete the
`text_search.lancedb` folder to let the app create a fresh database.
This requires indexing the TXT collection again. Keep a backup if you
need to preserve the old index.

## Recommended companion tool

For converting PDFs to TXT before searching, see [PDF
Parser](https://github.com/kalle07/pdf2txt-parser).

## Notes

-   Windows is the only platform tested so far.
-   The app searches plain `.txt` files; PDFs must be converted to text
    first.
-   You can copy and paste the extracted passages into your chat, notes
    or other tools.

<br>


<br>
I am not responsible for any errors or crashes on your system. If you use it, you take full responsibility!
