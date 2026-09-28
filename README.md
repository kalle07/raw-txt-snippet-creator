# raw-txt-snippet-creator with database
Actual version: 09-beta<br>
up to 3 keywords linked by <b>and</b>. Its like an embedder only with plain txt search!<br>
It's like opening a text editor, searching for a keyword, and finding X hits. Now the snippet extractor cuts out a section around each keyword and show it.
The maximum text found is never larger than the original text, as overlapping sections are merged!<br>
-> All is in characters<br>
-> Keep in mind 5000characters ~1200token (aprox one book page)
-> can handle large amount of data
-> will be indexed first
-> usual search need <100ms
- phrase search function will be implemented shortly.
- case-sensitive function will be implemented shortly.

Best in combination with my PDF Parser:
https://github.com/kalle07/pdf2txt-parser

# Hints
* Only windows tested!
* Only txt files, tested with several 2MB (several large books)
* Choose txt file or copy all txt files into sup folder "txt"
* Type one keyword or more
* Max_Distance_Chars - is the max distance between the keywords in characters
* Snippet_Conext_Chars - number of characters before and after the last word found
* Two search options "usual exact + wildcard" and "fuzzy-search"<br>
(wildcard search If you have the word “friendship” and search for “friend” it will not be found. You should use “friend*”. "?" is only one character like usual.)<br>
(fuzzy is sometime usefully , "1" means one characater replacement -> small typos / minor variations; fuzzy 2 most times found a lot of words)
* Now you can easily copy and paste to your chat


<img width="1482" height="1054" alt="grafik" src="https://github.com/user-attachments/assets/9a90f8d7-acc6-4bcd-b8b1-8a4f15ccfe56" />
<br>
<br>

📥 Downloads: <!--download-count-->010<!--/download-count-->
<br>
<br>


<br>
I am not responsible for any errors or crashes on your system. If you use it, you take full responsibility!
