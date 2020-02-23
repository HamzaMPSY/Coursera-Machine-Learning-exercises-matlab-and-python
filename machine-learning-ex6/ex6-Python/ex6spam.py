# Spam Classifier
import re # RegEx
from nltk.stem import PorterStemmer # Stemmer (Text Mining)
import numpy as np
from sklearn.svm import SVC
from scipy.io import loadmat




def processEmail(email,vocabList_dictionary):
	# Lower-casing
	email = email.lower()
	# Stripping HTML
	email = re.sub('<.*?>',' ',email)
	# Normalizing URLs:
	email = re.sub("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",'httpadrr',email)
	# Normalizing Emails:
	email = re.sub("\S+@\S+","emailaddr",email)
	# NOrmalizing Numbers:
	email = re.sub("\d+","number",email)
	# Dollars
	email = re.sub('[$]+','dollar',email)
	# Non_Word 
	email = re.sub("\W"," ",email)
	# white spaces 
	email = re.sub('[\t\n\r ]+',' ',re.sub('^ ','',email))

	ps = PorterStemmer()
	tokenList = [ps.stem(token) for token in email.split(" ")]
	email = ' '.join(tokenList)
	email = re.sub('^ ','',email)
	ret = []
	for token in email.split():
		if len(token) >1 and token in vocabList_dictionary:
			ret.append(vocabList_dictionary[token])

	return ret


def extractFeatures(vocabList_dictionary, word_indices):
	n = len(vocabList_dictionary)
	ret = np.zeros((n,1))

	for i in word_indices:
		ret[int(i)] = 1

	return ret



email = open("emailSample1.txt","r").read()
vocabList = open("vocab.txt","r").read()
vocabList = vocabList.split('\n')[:-1]

vocabList_dictionary = {}
for each in vocabList:
	value,key = each.split("\t")[:]
	vocabList_dictionary[key] = value
#word_indices = processEmail(email,vocabList_dictionary)
#print(np.sum(extractFeatures(vocabList_dictionary,word_indices)))

mat = loadmat("spamtrain.mat")
X = mat['X']
y = mat['y']
C=0.1
model = SVC(C=C,kernel = 'linear')
model.fit(X,y.ravel())
print("Training Accuracy:",(model.score(X,y.ravel()))*100,"%")

spam_mat_test = loadmat("spamTest.mat")
X_test = spam_mat_test["Xtest"]
y_test =spam_mat_test["ytest"]
print("Test Accuracy:",(model.score(X_test,y_test.ravel()))*100,"%")



