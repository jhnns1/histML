import sklearn
import pandas as pd
import numpy as np
from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import collections

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

import sys
import time
import os
import json

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import GRU
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Adadelta

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

import matplotlib.pyplot as plt

from contextlib import redirect_stdout      # print model.summary() in json

import os.path

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import time
import operator

starttime = time.time()

individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\KR_label_befund_vor_20180903.csv"
# individual_path_to_file = "/Users/Johannes/data.csv"
# individual_path_to_file = "/home/johannescheck/data.csv"
# individual_path_to_file = "/home/paulw/data.csv"

df = pd.read_csv(individual_path_to_file,
                  sep='$',
                  names=['label', 'befund'],
                  encoding='latin-1',
                  header=None)
# print("################ CHECK IMPORT ################")
# print(df.shape)
# print(df.head())
# print(df.describe())
# print(df.groupby('label').describe())
# Check for Null-Entries
# print(df.info())
# print(np.sum(df.isnull().any(axis=1)))
# print(df.isnull().any(axis=0))

# neues Feld mit Länge des Freitexts
df['length'] = df['label'].apply(len)
df['befund'].replace(to_replace='o.n.A.',value='onA',inplace=True,regex=True)
# df['befund'].replace(to_replace=',',value='',inplace=True,regex=True)
# df['befund'].replace(to_replace=';',value='',inplace=True,regex=True)
# df['befund'].replace(to_replace='-',value='',inplace=True,regex=True)

# Vorkommen der Label auf >= 100 beschränken
df['label_count'] = df['label'].value_counts()
# messages_ = df[df['label'].isin(counts[counts >= 10].index)]
stopwords_ = frozenset(['und','bei','da','einem','ein','einer','eine','vom','dem','mit','zum','in','im','cm','mm','am','an','um','es','auf','für','als','aus','eher','dabei','übrigen','übrige','üblicherweise','übersicht','übersandten','übersand','zumindest','zumeist','zu'])

# Train-Test-Split
# X_train, X_test, y_train, y_test = train_test_split(messages_['befund'], messages_['label'], test_size=0.0)
# X_train, X_validate_test, y_train, y_validate_test = train_test_split(messages_['befund'], messages_['label'], test_size=0.0, random_state=SEED, stratify=messages_['label'])
# X_validate, X_test, y_validate, y_test = train_test_split(X_validate_test, y_validate_test, test_size=0.5, random_state=SEED, stratify=y_validate_test)

text_clf_nb = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=(1,6))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', MultinomialNB()),])

# taken from https://medium.com/@cristhianboujon/how-to-list-the-most-common-words-from-text-corpus-using-scikit-learn-dad4d0cab41d
print(len(df))
df = df[df['label'] >= '8000/0']
df = df[df['label'] < '9993/0']
print(len(df))

X = df['befund'].values
y = df['label'].values

def get_frequent_words(n):
    """"
    List the top n words in a vocabulary according to occurence in a text corpus"""

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    dict = tokenizer.word_counts
    print(len(dict))

    sort_dict = sorted(dict.items(), key=lambda kv: kv[1],reverse=True)
    x = [i[0] for i in sort_dict][:n]
    y = [i[1] for i in sort_dict][:n]
    return x,y

n = 30
words, freq = get_frequent_words(n)

tmp = pd.DataFrame(freq)
print(tmp.describe())
bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

fig, ax = plt.subplots()
# fig.set_figwidth(10)
fig.set_figheight(15)
y_pos = np.arange(len(words))
#ax.yaxis.grid(True)
# plt.style.use('ggplot')
ax.grid(color='grey',linestyle=':',axis='x')
bars = plt.barh(y_pos,freq,facecolor=fg_color,alpha=1,edgecolor=bg_color)
#plt.bar(x,y,alpha=1)
# plt.xticks(rotation=90)
plt.xlabel('Häufigkeit')
plt.ylabel('Token aus Histologiebefund')
ax.set_yticks(y_pos)
ax.set_yticklabels(words)
xt = ax.get_xticks()
xt = xt[1:]
xt = np.append(xt,[1000])
ax.set_xticks(xt)

#plt.title('TOP {} medizinische Fachbegriffe' .format(n))
plt.subplots_adjust(left=0.35)
#plt.xlim(-0.5,)
plt.axis([0, 41000, -1, n])
ax.invert_yaxis()
ax.set_axisbelow(True)
ax.set_facecolor(bg_color)
plt.show()