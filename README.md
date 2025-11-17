# raw-txt-snippet-creator
Actual version: v08alpha<br>
Buzzword search with "AND" option within distance. Its like an embedder only with plain txt search!<br>
It's like opening a text editor, searching for a keyword, and finding X hits. Now the snippet extractor cuts out a section around each keyword and save.
The maximum text found is never larger than the original text, as overlapping sections are merged!<br>
-> all is in character and percent<br>
-> keep in mind 5000characters ~1200token (aprox one book page)

EXE on huggingface or relases(right side):<br>
https://huggingface.co/kalle07/raw-txt-snippet-creator

# Hints
* Only windows tested!
* Only txt files, tested with 2MB (one large book) ~10-20sec
* Choose one txt file or a whole folder
* Type one buzzword or more, only with AND (second search field) its connected with in a "distance option"
* snippet size and distance all in characters (5000 chars ~one book page, ~1400token)
* All matches found are cut out as a snippet (in % 0.3 before and 0.7 after the keyword)
* All overlaped snippets ar merged
* Two search options "usual exact + wildcard" and "fuzzy-search"<br>
(wildcard search If you have the word “friendship” and search for “friend” it will not be found. You should use “friend*”. "?" is only one character like usual.)<br>
(fuzzy is sometime usefully , but it dont work with any punctuation like ip adresses, but it can handle in some cases * and ?, in % I would not specify less than 80.)
* All snippets are appended and saved (one for wildcard one for fuzzy - file) in json format with te match and found position<br>
(the position you can see eg: in notepad++)
* first line also shows sum of all characters and estimated token
* Output files are always overwritten when you click “Search” again
* Now you can easily copy and paste to your chat


<img width="1557" height="1241" alt="grafik" src="https://github.com/user-attachments/assets/213f45e5-2219-48c9-bd49-b506d9199e5c" />
