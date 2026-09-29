# raw-txt-snippet-creator with database
Actual version: 09-beta<br>
up to 3 keywords linked by <b>and</b>. Its like an embedder only with plain txt search!<br>
It's like opening a text editor, searching for a keyword, and finding X hits. Now the snippet extractor cuts out a section around each keyword and show it.
The maximum text found is never larger than the original text, as overlapping sections are merged!<br>
-> All is in characters
-> Keep in mind 4000 characters ~1000token (aprox one book page)
-> Can handle large amount of data
-> Will be once indexed first, 10 big books need ~5-10sec
-> Load database at start < 1sec
-> Usual search need < 100ms
- Phrase search function will be implemented shortly.
- Case-sensitive function will be implemented shortly.

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
(fuzzy search is sometime usefully , "1" means one characater replacement -> small typos / minor variations; fuzzy 2 most times a lot of words are found)
* Now you can easily copy and paste to your chat


<img width="1482" height="1054" alt="grafik" src="https://github.com/user-attachments/assets/038d4041-0439-4c9f-ad53-d11ae8660eaa" />


<br>
download exe, no installation, direct working App (~150MB)
or
python -m venv venv
venv\Scripts\activate # On Windows
pip install -r requirements.txt
python start.py

<br>

📥 Downloads: <!--download-count-->010<!--/download-count-->
<br>
<br>


<br>
I am not responsible for any errors or crashes on your system. If you use it, you take full responsibility!
