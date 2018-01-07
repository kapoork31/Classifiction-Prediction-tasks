use as a word normalizer.

1. Essentially you give a txt file of words which you want to make sure are entered correctly when read by your system.
2. For example if you area creating a bot which works with specific key words, you want to be able to ensure user input has these words spelled correctly.
3. This solves this problem.
4. Firstly import the module.
5. call speller.makeTable(filename, floor, ceiling) to create a hash table of the words in your txt file that are indexed by length.
	floor defines the length of the shortest words allowed in your table which will be present at index 0, and ceiling defines largest word length which will present at index[-1]
	so for example a floor of 1 and a ceiling of 6 means the hash table will contain 6 lists, one at each index. hashtable[0] contains words of length 1, hashtbale[1] words of length 2 
	and so until hashtbale[5] that contain words of length 6. Following on from here another element is added to the hashtable at hashtbale[-1] which contains a list of all worlds longer than your defined ceiling.
	
6. Then call speller.spellCorrected(hashtable, string you want to be corrected ) and the ouput will be the corrected/normalized sentence. 