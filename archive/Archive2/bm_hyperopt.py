# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(13) # for reproducibility

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


import os
import csv
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
from keras.callbacks import EarlyStopping, Callback
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
from sklearn.metrics import log_loss, accuracy_score, fbeta_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

import functools
from keras import backend as K
import tensorflow as tf

# from scikitplot.metrics import plot_confusion_matrix

import nltk


# from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras import regularizers
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
from sklearn.metrics import confusion_matrix, fbeta_score, f1_score, precision_score, recall_score
# from textblob import TextBlob
import math

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
import datetime
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# customized
# from testcallback import TestCallback

import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder #encode classes

def config_(batch_size_, epochs_, lr_, dropout_, reg_):
    """Set hyperparameters"""
    min_count_per_class = 6  # Minimum W/ val_test 6, else 2
    max_features = 20000  # Maximum number of tokens in vocabulary
    max_len = 80  # Max Length for every sentence
    batch_size = batch_size_  # 32
    epochs = epochs_

    val_test_size = 0.3
    test_val_share = 0.667

    lr = lr_
    dropout = dropout_
    reg = reg_

    return min_count_per_class, max_features, max_len, batch_size, epochs, val_test_size, test_val_share, lr, dropout, reg


def load_data(n):
    """Load Data"""

    if n==0:
        df_ = pd.read_csv("/home/paulw/data.csv",
                          sep=';',
                          names=['label', 'befund'],
                          encoding='latin-1',
                          header=None)
    elif n==1:
        df_ = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\Data\\20180905-morph_code+morph_freitext.csv',
                          sep=';',
                          names=['label', 'befund'],
                          encoding='latin-1',
                          header=None)
    elif n==2:
        df_ = pd.read_csv('/Users/Johannes/Desktop/Laufwerk/00 Data/20180905-morph_code+morph_freitext.csv',
                          sep=';',
                          names=['label', 'befund'],
                          encoding='latin-1',
                          header=None)
    else:
        print('\nI did not recognize you. Seems like you should specify your path_to_file in bm_hypteropt/config_ Function!')
        sys.exit(0)
    print("\nData loaded.")

    return df_



def data_preprocessing(min_count_per_class, val_test_size, test_val_share, max_len, userid, show_basicstats=False, show_excludedstats=False, set_custom_n=False, custom_n=20):
    # Load data
    df_ = load_data(userid)

    print("\nData Preprocessing...")
    # To make sure that o.n.A. isn't torn apart by any tokenizer
    df_['befund'].replace(to_replace='o.n.A.',
                          value='onA',
                          inplace=True,
                          regex=True)

    if show_basicstats==True:
        print("\nShape of the dataset is: {}" .format(df_.shape))
        print("Number of classes given in the dataset: {}" . format(len(np.unique(df_['label']))))
        print("\nSome sanitary checks - looking for obvious hickups...")
        print(df_.head())
        print(df_.head(50))
        print("\nSome stats")
        print(df_.describe(include='all'))
        print(df_.info())

    # Only include labels with counts >= min_count_per_class to avoid train-test-split conflicts
    # And only include valid labels: >= 8000, <=9999
    counts = df_['label'].value_counts()
    df = df_[df_['label'].isin(counts[counts >= min_count_per_class].index)]
    # Kodierungsfehler werden idR durch >= min_count_per_class herausgefiltert
    df = df[df['label'] >= '8000/0']
    df = df[df['label'] < '9993/0']

    # Specialized stopwords, carefully chosen regarding the task
    stopwords_ = frozenset(['und', 'bei', 'da', 'einem', 'ein', 'einer', 'eine', 'vom',
                            'dem', 'mit', 'zum', 'in', 'im', 'cm', 'mm', 'am', 'an', 'um',
                            'es', 'auf', 'für', 'als', 'aus', 'eher', 'dabei'])

    if show_excludedstats==True:
        # Calculate excluded values
        missing_labels = set(df_['label']) - set(df['label'])
        print("Labels went missing bcoz fewer than {} occurences: {}" .format(min_count_per_class, missing_labels))
        print("Number of labels that went missing: {}" .format(len(missing_labels)))
        print("\nNumber of data entries lost: {}".format(len(df_) - len(df)))

    if set_custom_n==True:
        X = df['befund'][:custom_n].values
        print('\nLabel-Encoding...')
        # entweder use to_categorical + categorical_crossentropy
        # oder use just preprocessing + sparse_categorical_crossentropy
        y = to_categorical(preprocessing.LabelEncoder().fit_transform(df['label'][:custom_n].values), num_classes=None)
    else:
        X = df['befund'].values
        print('\nLabel-Encoding...')
        # entweder use to_categorical + categorical_crossentropy
        # oder use just preprocessing + sparse_categorical_crossentropy
        y = to_categorical(preprocessing.LabelEncoder().fit_transform(df['label'].values), num_classes=None)

    # store the current # of classes for use in output layer
    num_classes = y.shape[1]

    # # Train-Validate-Test-Split
    print('\nSplitting the data in train, validation and test set...')
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=val_test_size,
                                                                # stratify=y,
                                                                random_state=42,
                                                                shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_val_share,
                                                    # stratify=y_val_test,
                                                    random_state=42,
                                                    shuffle=True)

    # Preprocessing, Bag-Of-Words from keras.Tokenizer()
    print('\nPreprocessing of texts...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_val_tok = tokenizer.texts_to_sequences(X_val)
    X_test_tok = tokenizer.texts_to_sequences(X_test)

    print('\nPad seq (samples x time)...')
    X_train = sequence.pad_sequences(X_train_tok, max_len)
    X_val = sequence.pad_sequences(X_val_tok, max_len)
    X_test = sequence.pad_sequences(X_test_tok, max_len)

    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes

X_train, y_train, X_val, y_val, X_test, y_test, num_classes = data_preprocessing(6,0.3,0.667,80,userid=1)

def create_model(params):

    print("\nBuilding models...")
    model = Sequential()
    model.add(Embedding(input_dim=20000,output_dim=256))
    #model.add(BatchNormalization())
    model.add(GRU(256, activation=params['activation'], dropout=params['dropout'], recurrent_dropout=params['dropout'], kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(num_classes, activation="softmax"))

    # no lr decay for adam
    # "We propose Adam, a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients; the name Adam is derived from adaptive moment estimation."

    model.compile(loss="categorical_crossentropy",
                  optimizer=params['optimizer'],
                  metrics=['categorical_accuracy', 'fbeta_score', 'fmeasure','accuracy'])

    earlystopping = EarlyStopping(monitor='val_fbeta_score',
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode='auto',
                                  restore_best_weights=True)

    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_val, y_val),
                     batch_size=params['batch_size'],
                     epochs=50,
                     verbose=0,
                     shuffle=True,
                     callbacks=[earlystopping])

    score, cat_acc, fb, f1, acc = model.evaluate(X_test,
                                                 y_test,
                                                 batch_size=128)

    return {'loss': -acc, 'status': STATUS_OK}