import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import contractions
import string, unicodedata
from nltk.corpus import stopwords


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

#Making a dataframe
import pandas as pd
positive_frame = pd.DataFrame(p_reviews, columns=['reviews'])
negative_frame = pd.DataFrame(n_reviews, columns=['reviews'])

#Adding rating index to the dataframe
positive_frame['rating'] = 1
negative_frame['rating'] = 0

#merging positive and negative dataframes
data = [positive_frame, negative_frame]
reviews_data = pd.concat(data, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
          copy=True)

#shuffling rows
from sklearn.utils import shuffle
reviews_data = shuffle(reviews_data)

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
from tqdm import tqdm
corpus = []
for review in tqdm(reviews_data['reviews']):
    #Converting upper case to lower case
    review = review.lower()


    #removing all \n from the text
    review = review.replace('\n','')


    #Removing everything except a-z
    review = re.sub(r'[^a-z]',' ',review)


    #replacing contractions in the text
    review = contractions.fix(review)

    #tokenizing the review
    review = nltk.word_tokenize(review)

    #removing Stop words
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word, pos='v') for word in review if word not in set(stopwords.words('english'))]
    review = " ".join(review)
    corpus.append(review)

#Creating the Bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1,2))
features = tfidf.fit_transform(corpus)
X = pd.DataFrame(features.todense(),
            columns=tfidf.get_feature_names()
            )
y = reviews_data['rating'].values


#Building the machine learning model

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf_classifier.fit(X_train, y_train)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train.toarray(), y_train)

####################### ANN ########################################
# Importing the Keras libraries and packages
# Installing Tensorflow
#pip install tensorflow

# Installing Keras
#pip install --upgrade keras

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
ann_classifier = Sequential()

# Adding the input layer and the first hidden layer
ann_classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15056))

# Adding the second hidden layer
ann_classifier.add(Dense(output_dim = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
ann_classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
ann_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
ann_classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

################################## method using sklearn.pipeline ##########################################
#Splitting data
x_pipe_test = corpus[1500:2000]
y_pipe_test = reviews_data['rating'][1500:2000]

x_pipe_train = corpus[:1500]
y_pipe_train = reviews_data['rating'][:1500].values

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
])

text_clf.fit(x_pipe_train,y_pipe_train)
#########################################################################################################

# Predicting the Test set results
rf_pred = rf_classifier.predict(X_test)
nb_pred = nb_classifier.predict(X_test.toarray())

pipe_pred = text_clf.predict(x_pipe_test)

ann_pred = ann_classifier.predict(X_test)
ann_pred = (ann_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf = confusion_matrix(y_test, rf_pred)         #75.5%
cm_nb = confusion_matrix(y_test, nb_pred)         #73.5%

cm_pipe = confusion_matrix(y_pipe_test, pipe_pred)      #78.6

cm_ann = confusion_matrix(y_test, ann_pred)       #83.7%
