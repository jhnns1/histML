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
from keras.callbacks import TensorBoard

from gensim.models import Word2Vec

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

def config_(batch_size_, epochs_, lr_, dropout_, reg_):
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
        df_ = pd.read_csv('/Users/Johannes/20180905-morph_code+morph_freitext.csv',
                          sep=';',
                          names=['label', 'befund'],
                          encoding='latin-1',
                          header=None)
    else:
        print('\nI did not recognize you, seems like you should specify your path_to_file and output_directory!')
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

def create_model(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, epochs, max_features, embed_dim, gru_dim, den_dim, lr_, dropout_, reg_, userid, n_layer = 2, output_path=""):
    """

    """
    start_ = time.time()
    new_folder = time.strftime("%Y%m%d_%H%M")

    print("\nBuilding models...")
    model = Sequential()
    model.add(Embedding(input_dim=max_features,output_dim=embed_dim))
    #model.add(BatchNormalization())
    tmp = n_layer
    while tmp >= 2:
        model.add(GRU(gru_dim, dropout=dropout_, recurrent_dropout=dropout_, kernel_regularizer=regularizers.l2(reg_), return_sequences=True))
        tmp -= 1
    model.add(GRU(gru_dim, dropout=dropout_, recurrent_dropout=dropout_, kernel_regularizer=regularizers.l2(reg_)))
    model.add(Dense(den_dim, activation="softmax"))

    if userid==0:
        output_dir = "/home/paulw/model_output/" + new_folder + "/"
        modelcheckpoint = ModelCheckpoint(
            filepath=output_dir + 'weights-ep_{epoch:02d}-train_{fbeta_score:.2f}-val_{val_fbeta_score:.2f}.hdf5',
            monitor='val_fbeta_score',
            verbose=1,
            save_best_only=True,
            mode='max')

    elif userid==1:
        output_dir = "C:\\Users\\johannes.heck\\Google Drive\\Laufwerk\\99model_output\\" + new_folder + "\\"
        modelcheckpoint = ModelCheckpoint(filepath=output_dir + 'weights-ep_{epoch:02d}-train_{fbeta_score:.2f}-val_{val_fbeta_score:.2f}.hdf5',
                                          monitor='val_fbeta_score',
                                          verbose=1,
                                          save_best_only=True,
                                          mode='max')

    elif userid==2:
        output_dir = "/Users/Johannes/Desktop/model_output/" + new_folder + "/"
        modelcheckpoint = ModelCheckpoint(
            filepath=output_dir + 'weights-ep_{epoch:02d}-train_{fbeta_score:.2f}-val_{val_fbeta_score:.2f}.hdf5',
            monitor='val_fbeta_score',
            verbose=1,
            save_best_only=True,
            mode='max')

    elif output_path != "":
        output_dir = output_path + new_folder + "/"
        modelcheckpoint = ModelCheckpoint(
            filepath=output_dir + 'weights-ep_{epoch:02d}-train_{fbeta_score:.2f}-val_{val_fbeta_score:.2f}.hdf5',
            monitor='val_fbeta_score',
            verbose=1,
            save_best_only=True,
            mode='max')

    else:
        print("Kennen wir uns? Ich habe Dich leider nicht erkannt. Bitte übergebe der Funktion einen output_path mit Trenner für Windows ('\\') oder Unix ('/') am Ende.")
        sys.exit(0)

    if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # no lr decay for adam
    # "We propose Adam, a method for efficient stochastic optimization that only requires first-order gradients with little memory requirement. The method computes individual adaptive learning rates for different parameters from estimates of first and second moments of the gradients; the name Adam is derived from adaptive moment estimation."

    adam_ = optimizers.Adam(lr=lr_,
                            decay=0.0,
                            amsgrad=False,
                            clipnorm = 1.)

    model.compile(loss="categorical_crossentropy",
                  optimizer=adam_,
                  metrics=['categorical_accuracy', 'fbeta_score', 'fmeasure','accuracy'])

    earlystopping = EarlyStopping(monitor='val_loss',
                                  min_delta=0,
                                  patience=5,
                                  verbose=0,
                                  mode='min',
                                  restore_best_weights=True)

    log_dir = './logs_' + new_folder
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1,
                              write_graph=True, write_images=True)

    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_val, y_val),
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     shuffle=True,
                     callbacks=[modelcheckpoint,earlystopping,tensorboard])

    score, cat_acc, fb, f1, acc = model.evaluate(X_test,
                                                 y_test,
                                                 batch_size=batch_size)

    # score is the evaluation of the loss function for a given input
    print("\nTest score: {}" .format(score))
    print("Test categorical accuracy: {}" .format(cat_acc))
    print("Test fbeta measure: {}" .format(fb))
    print("Test f1 measure: {}" .format(f1))
    print("Test accuracy: {}" .format(acc))

    results = ([["Score",  score],["Categorical_Accuracy", cat_acc],["FBeta", fb],["F1", f1],["Accuracy",acc],["epochs",epochs],["n_layers",n_layer],["embed_dim",embed_dim],["GRU_dim",gru_dim],["Den_dim",den_dim],["lr",lr_],["dropout_",dropout_],["Reg",reg_]])
    output_sum = output_dir + "summary.csv"
    with open(output_sum, 'w') as fh:
        writer = csv.writer(fh, delimiter=' ', lineterminator='\n')
        writer.writerow(['Testresults/Params','&Stats'])
        for item in results:
            writer.writerow([item[0], item[1]])
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

    mins = (time.time() - start_) / 60
    h = (time.time() - start_) / 3600
    if h>1.0:
        print('\nPuh! Das ging jetzt über {} h'.format(h))
    else:
        print('\nPuh! Das ging jetzt über {} mins'.format(mins))


    # print(hist.history.keys())
    plt.figure("Accuracy, FBeta-Score")
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'], 'r')
    plt.plot(hist.history['fbeta_score'], 'g')
    plt.plot(hist.history['val_fbeta_score'], 'darkorange')
    plt.ylabel('accuracy, fbeta_score')
    plt.xlabel('epoch')
    plt.legend(['train_acc', 'validation_acc','train_fbeta','validation_fbeta'], loc='upper left')
    plt.text(0.02,
             0.0,
             "EP {}, N_Layer {}, EmbDim {}, GRUDim {}, DenDim {}, lr {}, dropout {}, reg {}" .format(epochs, n_layer, embed_dim, gru_dim, den_dim, lr_, dropout_, reg_),
             fontsize=8,
             transform=plt.gcf().transFigure)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir + "acc-val_acc-fbeta_score-val_fbeta_score.png")

    # plt.figure(200)
    # plt.plot(hist.history['categorical_accuracy'])
    # plt.plot(hist.history['val_categorical_accuracy'], 'r')
    # plt.ylabel('categorical accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')

    plt.figure("Loss")
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'], 'r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.text(0.02,
             0.0,
             "EP {}, N_Layer {}, EmbDim {}, GRUDim {}, DenDim {}, lr {}, dropout {}, reg {}".format(epochs, n_layer,
                                                                                                    embed_dim, gru_dim,
                                                                                                    den_dim, lr_,
                                                                                                    dropout_, reg_),
             fontsize=8, transform=plt.gcf().transFigure)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(output_dir + "loss-val_loss.png")

    # plt.figure(400)
    # plt.plot(hist.history['fbeta_score'])
    # plt.plot(hist.history['val_fbeta_score'], 'r')
    # plt.xlabel('epoch')
    # plt.ylabel('fbeta_score')
    # plt.legend(['train', 'validation'], loc='upper left')
    #
    # plt.figure(500)
    # plt.plot(hist.history['fmeasure'])
    # plt.plot(hist.history['val_fmeasure'], 'r')
    # plt.xlabel('epoch')
    # plt.ylabel('fmeasure_score')
    # plt.legend(['train', 'validation'], loc='upper left')

    #plt.show()

    return model