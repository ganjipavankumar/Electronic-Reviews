import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup
import re
import contractions
import string, unicodedata
from nltk.corpus import stopwords

from __future__ import print_function, division
from future.utils import iteritems
from builtins import range

############################## Getting Data #####################################################
#importing positive reviews
positive_reviews = BeautifulSoup(open('./data/samples/reviews/positive.review').read(),'lxml')
positive_reviews = positive_reviews.findAll('review_text')

#importing negative reviews
negative_reviews = BeautifulSoup(open('./data/samples/reviews/negative.review').read(),'lxml')
negative_reviews = negative_reviews.findAll('review_text')

#shuffling the data and equalizing to the number of negative reviews
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]

#extracting text from the positive_reviews formate
p_reviews = []
for review in positive_reviews:
    p_review = review.text
    p_reviews.append(p_review)

#extracting text from the positive_reviews formate
n_reviews = []
for review in negative_reviews:
    n_review = review.text
    n_reviews.append(n_review)

######################################### Data Preprocessing #######################################

def my_tokenize(p_review):

    #Converting upper case to lower case
    p_review = p_review.lower()

    #removing all \n from the text
    p_review = p_review.replace('\n','')

    #Removing everything except a-z
    p_review = re.sub(r'[^a-z]',' ',p_review)


    #replacing contractions in the text
    p_review = contractions.fix(p_review)

    #tokenizing the review
    p_review = nltk.word_tokenize(p_review)

    #removing non-ASCII characters from the list
    pos_review = []
    for word in p_review:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        pos_review.append(new_word)



    #removing Stop words
    word = [word for word in pos_review if not word in set(stopwords.words('english'))]
    pos_words = " ".join(word)
    pos_words = pos_words.split()


    #Lemmetizing the words
    lemmatizer = WordNetLemmatizer()
    pos_lemma = []
    for word in pos_words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        pos_lemma.append(lemma)


    return pos_lemma


word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in p_reviews:
    tokens = my_tokenizer(review)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in n_reviews:
    tokens = my_tokenizer(review)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

################################ Making array of zeros and ones ##############################
# now let's create our input matrices
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is for the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() # normalize it before setting label
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
# (N x D+1 matrix - keeping them together for now so we can shuffle more easily later
data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_to_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_to_vector(tokens, 0)
    data[i,:] = xy
    i += 1

###################### preparing training data and test data #################################
# shuffle the data and create train/test splits
# try it multiple times!
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be test
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

######################## Model Preparation ####################################################
#Logistic Regression
model = LogisticRegression()
model.fit(Xtrain, Ytrain)
print("Classification rate:", model.score(Xtest, Ytest))

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(Xtrain, Ytrain)

#Fitting model to ANN
####################### ANN ##########################
# Importing the Keras libraries and packages
# Installing Tensorflow
#!pip install tensorflow

# Installing Keras
#!pip install --upgrade keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
ann_classifier = Sequential()

# Adding the input layer and the first hidden layer
ann_classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 8547))

# Adding the second hidden layer
ann_classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
ann_classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
ann_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
ann_classifier.fit(Xtrain, Ytrain, batch_size = 10, nb_epoch = 20)

###################### Predicting the results ###############################################

# let's look at the weights for each word
# try it with different threshold values!
threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)

#predicting the results
nb_pred = nb_classifier.predict(Xtest)
ann_pred = ann_classifier.predict(Xtest)
ann_pred = (ann_pred > 0.5)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(Ytest, nb_pred)                 #98%
cm_ann = confusion_matrix(Ytest, ann_pred)               #100%
