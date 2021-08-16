import nltk
from nltk.tokenize import word_tokenize
sentence = "Books are on the table"

#Tokenization
words = word_tokenize(sentence)
#print(words)

#lowercasing
sentence = sentence.lower()
#tokenize
words = word_tokenize(sentence)
#print(words)

#stop words removal
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(sentence)

filtered_tokens = [w for w in word_tokens if w not in stop_words]
#print(filtered_tokens)

#lower-casing, tokenize, stop words removal
sentence = " Germany is in Europe. Europe is a continent. It is a developed country. There are many feet."
#lowercasing
sentence = sentence.lower()
word_tokens = word_tokenize(sentence)
print(word_tokens)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered_words = [w for w in word_tokens if w not in stop_words]
print(filtered_words)

#stemming - process of transforming a word to its root form
from nltk.stem import PorterStemmer, WordNetLemmatizer
ps = PorterStemmer()
stemmed_word_tokens = []
for w in filtered_words:
    stemmed_word_tokens.append(ps.stem(w))

print(stemmed_word_tokens)
nltk.download('wordnet')
#Lemmatization - Process of resolving a word to a lemma that exists in the dictionary
#lemmatization is always preferred over stemming
lemmatizer = WordNetLemmatizer()
lemmatized_words = []
for w in filtered_words:
    lemmatized_words.append(lemmatizer.lemmatize(w))

print(lemmatized_words)


