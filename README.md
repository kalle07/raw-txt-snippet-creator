# raw-txt-snippet-creator with database


https://github.com/kalle07/raw-txt-snippet-creator/releases/download/_v02/TextSearch-by-kalle07.exe

up to 3 keywords linked by <b>and</b> and you can seach within distance. Its like an embedder only with plain txt search!<br>
It's like opening a text editor, searching for a keyword, and finding X hits. Now the snippet extractor cuts out a section around each keyword and show it. This in advance can searc hfor several keywords in close range!<br>
Copy the EXE file to a folder of your choice. When you run it for the first time, the program will create the database and the TXT folder. You can now select text files or copy TXT files directly into the folder and "index new files".<br>
The maximum text found is never larger than the original text, as overlapping sections are merged!<br>
-> All is in characters
-> Keep in mind 4000 characters ~1000token (aprox one book page)
-> Can handle large amount of data (1000 txt files)
-> Will be once indexed first, (10 big books need ~5-10sec)
-> Load database at start < 1sec
-> Usual search need < 100ms
- Phrase search implemented
- Case-sensitive implemented
- document result browser implemented, right click save snippets

-> A new index should be created at new versions (delete folder "text_search.lancedb" to be safe), so there is no backward compatibility

Best in combination with my PDF Parser:
https://github.com/kalle07/pdf2txt-parser

# Hints
* Only windows tested!
* Only txt files, tested with several 2MB (several large books)
* Choose txt file or copy all txt files into sub folder "txt", you can also create  folders there
* Type one keyword or more
* Max_Distance_Chars - is the max distance between the keywords in characters
* Snippet_Conext_Chars - number of characters before and after the last word found for better content
* Two search options "usual exact/regex + wildcard" and "fuzzy-search"<br>
(wildcard search If you have the word “friendship” and search for “friend” it will not be found. You should use “friend*”. "?" is only one character like usual.)<br>
(fuzzy search is sometime usefully , "1" means one characater replacement -> small typos / minor variations; fuzzy 2 most times a lot of words are found)
* Now you can easily copy and paste to your chat


<img width="1482" height="1054" alt="grafik" src="https://github.com/user-attachments/assets/038d4041-0439-4c9f-ad53-d11ae8660eaa" />

<br>
<br>

download exe, no installation, direct working App (~150MB)<br>
or<br>
python -m venv venv<br>
venv\Scripts\activate # On Windows<br>
pip install -r requirements.txt<br>
python start.py<br>

<br>

📥 Downloads: <!--download-count-->016<!--/download-count-->
<br>
<br>


<br>
I am not responsible for any errors or crashes on your system. If you use it, you take full responsibility!
