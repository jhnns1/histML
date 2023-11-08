from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from collections import OrderedDict


messages = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\20180905-morph_code+morph_freitext.csv', sep=';',names=['label','morph_freitext'],encoding='latin-1')
# print("################ CHECK IMPORT ################")
# print(messages.shape)
# print(messages.head())
# print(messages.describe())
# print(messages.groupby('label').describe())
# print(messages.info())

# neues Feld mit Länge des Freitexts
messages['length'] = messages['morph_freitext'].apply(len)
messages['morph_freitext'].replace(to_replace='o.n.A.',value='onA',inplace=True,regex=True)
messages['token_length'] = [len(x.split(" ")) for x in messages['morph_freitext']]
print(max(messages['token_length']))

# Check for data balance
print(messages['label'].value_counts())



# Vorkommen der Label auf >= 100 beschränken

# ------------- ACHTUNG -------------
# für Vokabularzählung counts auf 0 und test_size = 0.0
# ----------------------------------

counts = messages['label'].value_counts()
messages_ = messages[messages['label'].isin(counts[counts >= 0].index)]

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(messages_['morph_freitext'], messages_['label'], test_size=0.0)

# print(X_train.shape)

# count = 0
# for i in range(len(X_train)):
#     if "o.n.A." in X_train[i]:
#         #print(X_train[i])
#         count += 1
#
# print(count)

# create an instance of CountVectorizer() class
count_vect = CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b")

# tokenize and build vocab
count_vect.fit(X_train,y_train)

# summarize
print(count_vect.vocabulary_)

vocab = count_vect.vocabulary_
print(len(vocab))
print(len(count_vect.get_feature_names()))


# encode document
bag_of_words = count_vect.transform(X_train)
sum_words = bag_of_words.sum(axis=0)

words_freq = [(word, sum_words[0, idx]) for word,idx in count_vect.vocabulary_.items()]
words_freq = sorted(words_freq, key= lambda x: x[1], reverse=True)

print(np.sum(sum_words))
# print(words_freq)

