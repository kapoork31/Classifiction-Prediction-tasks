import pandas as pd
import string
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd



def load(fileName,yourName,theirName): # function to read in the text file and format it into a suitable structured format
    with open(fileName, "r", encoding="utf8") as words_file:
        words = []
        b = ""
        for line in words_file:
            word = line.strip()
            words.append(word) # simply reading text file into a list called words.
    
    messages = []
    for line in words:
        if yourName in line or theirName in line:
            messages.append(line)
            messagesSplit = [] # this loop is used to extract messages which contain your name or your contact's name. Essentially 
            # to ensure the sender is either yourself or your contact.
            
    names = []
    messgs = []
    for message in messages:
        if(": " in message):
            important = message.split("-") # split the string by : needed to get the message
            dets = important[1].split(':') 
            name = dets[0] # name of sender
            names.append(name)
            messg = dets[1] # message sent
            messgs.append(messg)
            
    df = pd.DataFrame(
    {'names': names,
     'message': messgs
    })
    
    return (df) # return dataframe with message and name of sender
	
df1 = load("txt file of 1st conversation",'your name','contact 1 name') # df of first contact
df2 = load("txt file of 2nd converstaion",'your name',"contact 2 name")  # df of second contact

c1 = df1[df1.names != "contact 1 name"] # extract df where sender is yourself
c2 = df2[df2.names != "contact 2 name"] # extract df where sender is yourself

c1['R'] = "contact 1 name" # reciever name, used as label
c2['R'] = "contact 2 name" # recieve name, used as label
	
	
if(len(c1) > len(c2)):
	frames = [c1.sample(len(c2)),c2]# in my case c1 is bigger than c2. So to balance the dataset I take a subsample of c1 to match the size of c2.
	result = pd.concat(frames) # concat the two contact data frames 

else:
	frames = [c1,c2.sample(len(c1))]# c2 could be bigger than c1. So to balance the dataset I take a subsample of c2 to match the size of c1.
	result = pd.concat(frames) # concat the two contact data frames 

data = list(result.message) # take the messages
target = list(result.R) # target = reciever
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=0) # training and test split
	
vectorizer = CountVectorizer(stop_words = 'english')
#-Compute term-document matrix
td_train = vectorizer.fit_transform( X_train)
# fill term-document matrix
td_test = vectorizer.transform( X_test)# create term-document matrix of the test 


clfNV = MultinomialNB().fit(td_train,y_train) # bayes classifier trained
clfnn = MLPClassifier().fit(td_train,y_train) # mutli layer perceptron trained

predNB = clfNV.predict(td_test) # predict test set
prednn = clfnn.predict(td_test) # predict test set

print ('Accuracy score for bayes = %f' % metrics.accuracy_score(y_test, predNB, normalize=True))
print ('Accuracy score for nn = %f' % metrics.accuracy_score(y_test, prednn, normalize=True))
# precision for one contact was 0.7 and 0.56 for the other
# recall for one contact was 0.79 and 0.48 for the other


	
