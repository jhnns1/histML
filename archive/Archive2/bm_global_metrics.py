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
from sklearn.metrics import confusion_matrix, fbeta_score, f1_score, precision_score, recall_score, precision_recall_fscore_support, roc_auc_score
# from textblob import TextBlob
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from gensim.models import Word2Vec

from keras.callbacks import TensorBoard

import logging
import time
from sys import platform
import sys
import datetime

# customized
# from testcallback import TestCallback

import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder #encode classes


class Metrics_(Callback):
    #taken from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    def on_train_begin(self, logs={}):
        #self.val_fbs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_rocauc = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        # val_predict_roc = np.argmax(val_predict, axis=1)
        # val_targ_roc = np.argmax(val_targ, axis=1)

        _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(val_targ, val_predict, beta=1.0, average='weighted')
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('val_f1: {}, val_precision: {}, val_recall {}' .format(_val_f1, _val_precision, _val_recall))
        #print('val_f1: {}, val_precision: {}, val_recall {}'.format(_val_f1, _val_precision, _val_recall))
        return


metrics_ = Metrics_()



def set_hyper(batch_size_, epochs_, lr_, dropout_, reg_):
    """Set hyperparameters"""
    min_count_per_class = 6  # Minimum W/ val_test 6, else 2
    max_features = 20000  # Maximum number of tokens in vocabulary
    max_len = 80  # Max Length for every sentence
    batch_size = batch_size_  # 32
    epochs = epochs_

    val_test_size = 0.3
    test_val_share = 0.5

    lr = lr_
    dropout = dropout_
    reg = reg_
    return min_count_per_class, max_features, max_len, batch_size, epochs, val_test_size, test_val_share, lr, dropout, reg


def load_data():
    df_ = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\Data\\20180905-morph_code+morph_freitext.csv',
                          sep=';',
                          names=['label', 'befund'],
                          encoding='latin-1',
                          header=None)

    return df_

def data_preprocessing(min_count_per_class, val_test_size, test_val_share, max_len, show_basicstats=False, show_excludedstats=False, set_custom_n=False, custom_n=20):
    # Load data
    df_ = load_data()

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

def create_model(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, epochs, max_features, embed_dim, gru_dim, den_dim, lr_, dropout_, reg_, verbose=2):
    """

    """

    start_ = time.time()
    new_folder = time.strftime("%Y%m%d_%H%M")

    log_dir = './logs_' + new_folder
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1,
                              write_graph=True, write_images=True)

    print("\nBuilding models...")
    model = Sequential()
    model.add(Embedding(input_dim=max_features,output_dim=embed_dim))
    #model.add(BatchNormalization())
    model.add(GRU(gru_dim, dropout=dropout_, recurrent_dropout=dropout_, kernel_regularizer=regularizers.l2(reg_), return_sequences=True))
    model.add(GRU(gru_dim, dropout=dropout_, recurrent_dropout=dropout_, kernel_regularizer=regularizers.l2(reg_)))
    model.add(Dense(den_dim, activation="softmax"))

    # no lr decay for adam
    # "We propose Adam, a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients; the name Adam is derived from adaptive moment estimation."

    adam_ = optimizers.Adam(lr=lr_, decay=0.0, amsgrad=False)

    model.compile(loss="categorical_crossentropy",
                  optimizer=adam_,
                  metrics=['fbeta_score','precision','recall','accuracy'])
                  #weighted_metrics=['fbeta_score','precision','recall','accuracy'])

    earlystopping = EarlyStopping(monitor='val_fbeta_score',
                                  min_delta=0,
                                  patience=10,
                                  verbose=0,
                                  mode='auto',
                                  restore_best_weights=True)

    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_val, y_val),
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=verbose,
                     shuffle=True,
                     callbacks=[metrics_, earlystopping, tensorboard])

    f1s = metrics_.val_f1s
    pres = metrics_.val_precisions
    rec = metrics_.val_recalls
    rocauc = metrics_.val_rocauc

    print(metrics_.val_f1s)
    print(metrics_.val_precisions)
    print(metrics_.val_recalls)
    print(metrics_.val_rocauc)

    preds = model.predict_classes(X_test)
    for i in range(len(X_test)):
        print("X {}, predicted {}, true {}" .format(X_test[i], preds[i], y_test[i]))

    print(hist.history.keys())

    mins = (time.time() - start_) / 60
    h = (time.time() - start_) / 3600
    if h>1.0:
        print('\nPuh! Das ging jetzt über {} h'.format(h))
    else:
        print('\nPuh! Das ging jetzt über {} mins'.format(mins))

    return model