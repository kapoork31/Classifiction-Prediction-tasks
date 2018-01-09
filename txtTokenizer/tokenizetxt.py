#tokenizetxt

def tokenizetxt(fileName, sep):

	words = []
	if(sep == 1):
		with open(fileName, "r") as words_file:
			for line in words_file:
				t = line.split(',')
				word = map(lambda s: s.strip(), t)
				words.extend(word) # simply reading text file into a list called words.
	
	if(sep == 0):
		with open(fileName, "r") as words_file:
			for line in words_file:
				nothing = [',','.',')','!','?',':','"']
				space = ['(','%','/']
				string = line
				for ch in nothing:
					string = string.replace(ch,'')
					
				for ch in space:
					string = string.replace(ch,' ')
				
				word = string.split(' ')
				word = list(map(lambda s: s.strip(), word))
				word = list(filter(None,word))
				words.extend(word) # simply reading text file into a list called words.
				
	return words
	
words = tokenizetxt('readme.txt',0)
print (words)

