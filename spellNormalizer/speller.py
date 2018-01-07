

def makeTable(fileName,floor,ceiling):
    
    def greaterThan_size(words,size):
        return [word for word in words if len(word) > size] # end case, last element in hash table to hold words > ceiling length

    def by_size(words, size):
        return [word for word in words if len(word) == size] # return list of words form a list of words equal to some length

    def createHash(fileName,floor,ceiling): # create hash table of words based on word length, floor will be the length of words in 
        # first index of the list, to make it simple make this 1, ceiling is length of words in 2nd last element of the list
        
        with open(fileName, "r") as words_file:
            words = []
            for line in words_file:
                t = line.split(',')
                word = map(lambda s: s.strip(), t)
                words.extend(word) # simply reading text file into a list called words.
                
        table = []
        for x in range(floor,ceiling):
            table.append(by_size(words,x)) # append the list o fwords of a certain length to the table
        table.append(greaterThan_size(words,ceiling-1)) # at the last element, add list of words whos lengths are > ceiling

        return table
    
    return createHash(fileName,floor,ceiling)

#table = makeTable('1-1000.txt',1,11) # 1-1000.txt included. txt files with csv data or newline seperated date should work.
# this above line is needed to create the table/hashtable passed to the spellerCorrected function	

def spellerCorrected(table,word):
    
    def correction(word): 
        if(word == candidates(word)):
                return word # if the word exists in the candidate list return the word
        voc =  list(candidates(word)) 
        match = [len(set(x).intersection(word)) for x in voc] # match the entered word with the word with most matching letters in the candidate list
        return voc[match.index(max(match))]

    def candidates(word): 
        # Generate possible spelling corrections for word
        speller = table # load in the hash table
        length = len(word)
        # set up to search words of length -3 to length + 3 of the given word
        floor = (length - 3) - 1 # floor is length -4 as x[1:4] in python will not include the x[1]
        if(floor<0): # if the floor of the word is < 0 then set the floor to 0 to avoid an out of bound exception
            floor = 0
            
        ceiling = (length + 3) 
        
        if(ceiling > len(speller)): # if celing > len(hashtable)
            ceiling = len(speller) # set ceiling to be last element in list
            floor = len(speller) - 3  -1 # 
            
        x = speller[floor:ceiling]
        candidates = []
        for i in x:
            for w in i:
                candidates.append(w) # obtain words from the table within the range of floor and ceiling

        if (word in candidates):
            return word # if the words exists just return the word

        return candidates # return the candidate words
    def fullCorrect(s):
        sentence = s.lower().split() # take in the string
        corrected = [correction(word) for word in sentence] # get matching words
        full = ' '.join(word for word in corrected) # rejoin the matching words into the sentencce
        return full # return the sentence
    return fullCorrect(word) # print out the corrected word.
	
	
#print(spellerCorrected(table,'realy'))
# test output, output for this is early as really is not in the 1-1000.txt file.