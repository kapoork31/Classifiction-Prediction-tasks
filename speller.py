import pickle
import os
parent = os.path.dirname(os.getcwd())


def correction(word): 
        if(word == candidates(word)):
            return word 
        voc =  list(candidates(word))
        match = [len(set(x).intersection(word)) for x in voc]
        return voc[match.index(max(match))]

def candidates(word): 
        "Generate possible spelling corrections for word."
        #return (known([word]) or (known(edits1(word)).union(known(edits2(word)))).union(known(edits3(word))) or [word])
        speller = pickle.load( open(parent + '\\resources\\models\\spellcorrectHashtable.p') )
        length = len(word)
        floor = (length - 3) - 1
        if(floor<0):
            floor = 0
        ceiling = (length + 3) 
        if(ceiling > len(speller)):
            ceiling = len(speller)
            floor = len(speller) - 3
        x = speller[floor:ceiling]
        candidates = []
        for i in x:
            for w in i:
                candidates.append(w)

        if (word in candidates):
            return word
        print candidates
        return candidates
def fullCorrect(s):
        sentence = s.lower().split()
        corrected = [correction(word) for word in sentence]
        full = ' '.join(word for word in corrected)
        return full
        