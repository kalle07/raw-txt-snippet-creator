# raw-txt-snippet-creator
Buzzword search with "AND" option within distance. Its like an embedder only with plain txt search!
The maximum text found is never larger than the original text, as overlapping sections are merged!

# Hints
* Only windows tested!
* Only txt files
* Choose one txt file or a whole folder
* Type a buzzword or more, with AND its connected with a distance.
* All matches found are cut out as a snippet.
* All overlaped snippets ar merged
* Two search options "usual wildcard" and "fuzzy-search"
(wildcard search If you have the word “friendship” and search for “friend” it will not be found. You should use “friend*”. "?" is only one character like usual.)
(fuzzy is sometime usefully , but it dont work with any punctuation like ip adresses, but it can handle in some cases * and ?)
* All snippets are appended and saved (one for wildcard one for fuzzy) in json format with te match and found position
* Output files are always overwritten when you click “Search” again.
* Now you can easily copy and paste to your chat


<img width="741" height="620" alt="grafik" src="https://github.com/user-attachments/assets/80d9ea12-68ca-4297-8641-1055cdfc91a3" />
