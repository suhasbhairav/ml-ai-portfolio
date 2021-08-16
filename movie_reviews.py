import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
import pickle


def clean_text(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned, '', text)

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    return text.lower()

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_text(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

data = pd.read_csv('./datasets/imdb.csv')
"""
print(data.shape)
print(data.head())
print(data.info())
print(data.sentiment.value_counts())
"""
data.sentiment.replace('positive', 1, inplace=True)
data.sentiment.replace('negative', 0, inplace=True)

#print(data.head(10))
data.review = data.review.apply(clean_text)
#print(data.review[0])

data.review = data.review.apply(is_special)
data.review = data.review.apply(to_lower)
data.review = data.review.apply(remove_stop_words)
data.review = data.review.apply(stem_text)

#create bag of words
X = np.array(data.iloc[:, 0].values)
y = np.array(data.sentiment.values)
cv = CountVectorizer(max_features=1000)
X = cv.fit_transform(data.review).toarray()
#print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
clf = GaussianNB()
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))

clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
