# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(13) # for reproducibility

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


import os
import codecs
import theano
# import jellyfish
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
from keras import optimizers
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

# from scikitplot.metrics import plot_confusion_matrix

import nltk


# from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import RMSprop, Adam, Nadam
from keras.activations import relu, elu, tanh

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
# from textblob import TextBlob
import math

from keras.losses import categorical_crossentropy, logcosh
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from gensim.models import Word2Vec

import logging
import time
from sys import platform
import sys

# customized
# from testcallback import TestCallback

import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder #encode classes

# Customizing
# Use AVX2 (for windows): https://github.com/fo40225/tensorflow-windows-wheel, install via pip install <filename>
# Mac: https://github.com/lakshayg/tensorflow-build

plt.rcParams['font.sans-serif'] = "Arial"
bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

# Establish logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Parameters
min_count_per_class = 6         # Minimum W/ val_test 6, else 2
max_features = 20000            # Maximum number of tokens in vocabulary
max_len = 80                    # Max Length for every sentence
batch_size = 32                # 32

val_test_size = 0.3
test_size = 0.3
test_val_share = 0.667

start_ = time.time()

print('Loading data...')

if platform =="win32" or platform=="cygwin":
    df_ = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\Data\\20180905-morph_code+morph_freitext.csv',
                      sep=';',
                      names=['label','befund'],
                      encoding='latin-1',
                      header=None)
elif platform=="darwin":
    df_ = pd.read_csv('/Users/Johannes/Desktop/Laufwerk/00 Data/20180905-morph_code+morph_freitext.csv',
                      sep=';',
                      names=['label','befund'],
                      encoding='latin-1',
                      header=None)
else:
    print('\nCould not recognize the operating platform')
    sys.exit(0)

df_['befund'].replace(to_replace='o.n.A.',
                      value='onA',
                      inplace=True,
                      regex=True) # To make sure that o.n.A. isn't torn apart by any tokenizer

# print("\nShape of the dataset is: {}" .format(df_.shape))
# print("Number of classes given in the dataset: {}" . format(len(np.unique(df_['label']))))
# print("\nSome sanitary checks - looking for obvious hickups...")
# print(df_.head())
# print(df_.head(50))
# print("\nSome stats")
# print(df_.describe(include='all'))
# print(df_.info())

# Only include labels with counts >= min_count_per_class to avoid train-test-split conflicts
# And only include valid labels: >= 8000, <=9999
counts = df_['label'].value_counts()
df = df_[df_['label'].isin(counts[counts >= min_count_per_class].index)]
# Kodierungsfehler werden idR durch >= min_count_per_class herausgefiltert
df = df[df['label']>='8000/0']
df = df[df['label']<'9993/0']

# Specialized stopwords, carefully chosen regarding the task
stopwords_ = frozenset(['und','bei','da','einem','ein','einer','eine','vom',
                        'dem','mit','zum','in','im','cm','mm','am','an','um',
                        'es','auf','für','als','aus','eher','dabei'])

# # Calculate excluded values
# missing_labels = set(df_['label']) - set(df['label'])
# print("Labels went missing bcoz fewer than {] occurences: {}" .format(min_count_per_class, missing_labels))
# print("Number of labels that went missing: {}" .format(len(missing_labels)))
print("\nNumber of data entries lost: {}" .format(len(df_) - len(df)))

X = df['befund'][:20].values
print('\nLabel-Encoding...')

# entweder use to_categorical + categorical_crossentropy
# oder use just preprocessing + sparse_categorical_crossentropy
y = to_categorical(preprocessing.LabelEncoder().fit_transform(df['label'][:20].values), num_classes=None)
num_classes = y.shape[1]        # store the current # of classes for use in output layer

# # Train-Validate-Test-Split
print('\nSplitting the data in train, validation and test set...')
X_train, X_val_test, y_train, y_val_test = train_test_split(X,y,test_size=val_test_size,
                                                            #stratify=y,
                                                            random_state=42,
                                                            shuffle=False)
X_val, X_test, y_val, y_test = train_test_split(X_val_test,y_val_test,test_size=test_val_share,
                                                #stratify=y_val_test,
                                                random_state=42,
                                                shuffle=False)

# Preprocessing, Bag-Of-Words from keras.Tokenizer()
print('\nPreprocessing of texts...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_train_tok = tokenizer.texts_to_sequences(X_train)
X_val_tok = tokenizer.texts_to_sequences(X_val)
X_test_tok = tokenizer.texts_to_sequences(X_test)

print('\nPad seq (samples x time)...')
X_train = sequence.pad_sequences(X_train_tok,max_len)
X_val = sequence.pad_sequences(X_val_tok,max_len)
X_test = sequence.pad_sequences(X_test_tok,max_len)

X = sequence.pad_sequences(tokenizer.texts_to_sequences(X),max_len)

print('\nBuilding models...')
# input shape?
# https://keras.io/getting-started/sequential-model-guide/#specifying-the-input-shape
# input_shape=(timesteps, data_dim), timesteps = 8, data_dim = 16

# Vanilla GRU

def opt_my_model(a,b,c,d,params):
    model = Sequential()
    model.add(Embedding(max_features, 256))             # turns positive vectors into dense vectors (a la word2vec)
    model.add((GRU(256,activation=params['activation'])))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=params['optimizer'](),
                  metrics=['categorical_accuracy','fbeta_score','fmeasure'])

    print('\nTrain...')

    out = model.fit(X_train,
                     y_train,
                      validation_data=(X_val,y_val),
                      batch_size=params['batch_size'],
                      epochs=params['epochs'],
                     verbose=2)

    return out, model

p = {'lr': (0.0001, 0.001, 0.5, 2, 10, 30),
     'first_neuron':[4, 8, 16, 32, 64, 128],
     'hidden_layers':[2,3,4,5,6],
     'batch_size': [2, 3, 4,16,32,128],
     'epochs': [100,200,300],
     'dropout': (0, 0.40, 10),
     'weight_regulizer':[None],
     'emb_output_dims': [None],
     'optimizer': [RMSprop, Nadam],
     'losses': [categorical_crossentropy, logcosh],
     'activation':[relu, elu, tanh]}

import talos as ta

h = ta.Scan(x=X,
            y=y,
            model=opt_my_model,
            grid_downsample=0.5,
            params=p,
            experiment_no='1')
#
# score, cat_acc, fb, f1 = model.evaluate(X_test,
#                                              y_test,
#                                              batch_size=batch_size)
#
# print("\nTest score:", score)
# print("Test categorical accuracy:", cat_acc)
# print("Test fbeta measure:", fb)
# print("Test fbeta measure:", f1)
#
# print('\nPuh! Das ging jetzt über {} mins' .format((time.time() - start_)/60))
#
# print('\nPlotting Acc + Loss...')
# # Access via hist.history.keys() => ['acc','loss','val_acc','val_loss']
# print(hist.history.keys())
