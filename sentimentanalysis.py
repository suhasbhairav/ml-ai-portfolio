import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import CountVectorizer


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + \
           ' '.join(emojis).replace('-', '')
    return text

count = CountVectorizer()
data = pd.read_csv('./datasets/sentimentanalysis/Train.csv')
#print(data.head())

import re

data['text']=data['text'].apply(preprocessor)
print(data['text'])

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('wordnet')
def tokenizer_lemmatize(text):
    return [lemmatizer.lemmatize(w) for w in text.split()]

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None,
                        tokenizer=tokenizer_lemmatize,
                        use_idf=True,
                        smooth_idf=True)

x = tfidf.fit_transform(data.text)
y = data.label.values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8)

from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=6, scoring='accuracy', random_state=0, n_jobs=-1, verbose=3, max_iter=500)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
