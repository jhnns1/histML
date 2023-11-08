# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(1337) # for reproducibility

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)


import os
import codecs
import theano
import jellyfish
import gc
import itertools
import pandas as pd
import collections as col
from collections import Counter
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Masking
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn.model_selection import StratifiedKFold
from nltk import tokenize
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
# from attention import AttLayer

# ___________________________

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

from scikitplot.metrics import plot_confusion_matrix

import nltk


from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam

# ____________________________________________




import sklearn

import matplotlib.pyplot as plt
from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from collections import OrderedDict
from textblob import TextBlob
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from gensim.models import Word2Vec

import logging

import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder #encode classes

plt.rcParams['font.sans-serif'] = "Arial"
bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

# Establish logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Parameters
min_count_per_class = 6         # W/ val_test 6, else 2
max_features = 20000            # Maximum number of tokens in vocabulary
max_len = 50                    # Max Length for every sentence
val_test_size = 0.3
test_size = 0.3
test_val_share = 0.667



print('Loading data...')
df_ = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\Data\\20180905-morph_code+morph_freitext.csv', sep=';',names=['label','befund'],encoding='latin-1',header=None)
df_['befund'].replace(to_replace='o.n.A.',value='onA',inplace=True,regex=True) # To make sure that o.n.A. isn't torn apart by any tokenizer

# print("\nShape of the dataset is: {}" .format(df_.shape))
# print("Number of classes given in the dataset: {}" . format(len(np.unique(df_['label']))))
# print("\nSome sanitary checks - looking for obvious hickups...")
# print(df_.head())
# print(df_.head(50))
# print("\nSome stats")
# print(df_.describe(include='all'))
# print(df_.info())

# Only include labels with counts >= min_count_per_class to avoid train-test-split conflicts
counts = df_['label'].value_counts()
df = df_[df_['label'].isin(counts[counts >= min_count_per_class].index)]
stopwords_ = frozenset(['und','bei','da','einem','ein','einer','eine','vom','dem','mit','zum','in','im','cm','mm','am','an','um','es','auf','für','als','aus','eher','dabei']) # Specialized stopwords, carefully chosen regarding the task

# # Calculate excluded values
# missing_labels = set(df_['label']) - set(df['label'])
# print("Labels went missing bcoz fewer than {] occurences: {}" .format(min_count_per_class, missing_labels))
# print("Number of labels that went missing: {}" .format(len(missing_labels)))

# Pandas Series obj to preserve values and index
X= df['befund'].values
y = df['label'].values


# # Train-Validate-Test-Split
print('\nSplitting the data in train and test set...')
# X_train, X_val_test, y_train, y_val_test = train_test_split(X,y,test_size=val_test_size,stratify=y,random_state=42,shuffle=True)
# X_val, X_test, y_val, y_test = train_test_split(X_val_test,y_val_test,test_size=test_val_share,stratify=y_val_test,random_state=42,shuffle=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=42,shuffle=True)


# tokenizer = Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(X_train)
# print("\nFound {} unique tokens in the training set." .format(len(tokenizer.word_index)))
# print("Found {} tokens in total in the training set." .format(sum(tokenizer.word_counts.values())))
# word_count = [k[0] for k in tokenizer.word_counts.items() if k[1] > 1] # Just take items that occure more than once
# word_index = tokenizer.word_index
# word_keys = list(word_index.keys())
#
# # print(OrderedDict(sorted(tokenizer.word_counts.items(), key= lambda x: x[1], reverse=True)))
#
# # Delete words with freq < 2
# for i in range(len(word_keys)):
#     if word_keys[i] not in word_count:
#         # print([word_keys[i]])
#         del word_index[word_keys[i]]
#
# # Reset tokenizer.word_index
# tokenizer.word_index = word_index
#
# # print("\nSaving variables...")
# # np.save('DICT.npy', word_index)

