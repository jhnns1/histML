# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


import numpy as np
import pandas as pd
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
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback, TensorBoard, ModelCheckpoint

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from hyperopt import STATUS_OK

import matplotlib.pyplot as plt

from pprint import pprint

from contextlib import redirect_stdout

#####################################
# Customization
#####################################

class Metrics_(Callback):
    #taken from https://medium.com/@thongonary/how-to-compute-f1-score-for-each-epoch-in-keras-a1acd17715a2
    def on_train_begin(self, logs={}):
        #self.val_fbs = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_rocauc = []
        self.val_fbs = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]

        # val_predict_roc = np.argmax(val_predict, axis=1)
        # val_targ_roc = np.argmax(val_targ, axis=1)

        _val_precision, _val_recall, _val_f1, _ = precision_recall_fscore_support(val_targ, val_predict, beta=1.0, average='weighted')
        _val_fb = fbeta_score(val_targ, val_predict, beta = 3.0, average='weighted')
        self.val_f1s.append(_val_f1)
        self.val_fbs.append(_val_fb)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        #print('val_f1: {}, val_fb: {}, val_precision: {}, val_recall {}' .format(_val_f1, _val_fb, _val_precision, _val_recall))
        return

metrics_ = Metrics_()

# Custom metrics for EarlyStopping, taken from keras 1.0
# just batch-wise evaluation!
def precision_(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score_(y_true, y_pred, beta=2):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision_(y_true, y_pred)
    r = recall_(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure_(y_true, y_pred):
    return fbeta_score_(y_true, y_pred, beta=1)

#####################################
# Model
#####################################

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

def load_data():
    """Load Data"""

    individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\20180905-morph_code+morph_freitext.csv"
    #individual_path_to_file = "/Users/Johannes/data.csv"
    #individual_path_to_file = "/home/johannescheck/data.csv"

    df_ = pd.read_csv(individual_path_to_file,
                      sep=';',
                      names=['label', 'befund'],
                      encoding='latin-1',
                      header=None)

    print("\nData loaded.")

    return df_


    # Load data

def data_preprocessing(min_count_per_class,
                       val_test_size,
                       test_val_share,
                       max_len,
                       show_basicstats=False,
                       show_excludedstats=False,
                       set_custom_n=False,
                       custom_n=20):
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
        le = preprocessing.LabelEncoder().fit_transform(df['label'][:custom_n].values)
        # reverse_y_label = list(le.inverse_transform(df['label'][:custom_n].values))
        # print(le.classes_)
        y = to_categorical(le, num_classes=None)
        # reverse_y_to_cat = np.argmax(y, axis=1)
        # print(reverse_y_to_cat)

    else:
        X = df['befund'].values
        print('\nLabel-Encoding...')
        # entweder use to_categorical + categorical_crossentropy
        # oder use just preprocessing + sparse_categorical_crossentropy
        le = preprocessing.LabelEncoder().fit_transform(df['label'].values)
        # reverse_y_label = list(le.inverse_transform(df['label'].values))
        # print(le.classes_)
        y = to_categorical(le, num_classes=None)
        # reverse_y_to_cat = np.argmax(y, axis=1)
        # print(reverse_y_to_cat)

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
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
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

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, reverse_word_map

def create_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, reverse_word_map):

    start_ = time.time()
    new_folder = time.strftime("%Y%m%d_%H%M")
    #log_dir = './logs_' + new_folder
    log_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\logs_"
    #log_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\logs_\\final\\{}".format(time())
    tensorboard = TensorBoard(log_dir=log_dir,
                              histogram_freq=1,         # plot the distributions of weights and biases in the nn
                              write_graph=True,         # print the graph of nn as defined internally
                              write_images=True)        # create an image by combining the weight of nn

    print("\nBuilding model...")
    model = Sequential()
    model.add(Embedding(input_dim=20000,output_dim=128))
    # model.add(GRU(256,
    #               dropout=0.4,
    #               recurrent_dropout=0.4,
    #               kernel_regularizer=regularizers.l2(0.02),
    #               return_sequences=True,
    #               activation=None))
    # model.add(Activation(activation='tanh'))
    # model.add(Dropout())
    # model.add(BatchNormalization())

    model.add(GRU(128,
                  dropout=0.257,
                  recurrent_dropout=0.257,
                  kernel_regularizer=regularizers.l2(0.33),
                  activation='tanh'))
    # model.add(Activation(activation='tanh'))
    # model.add(Dropout())
    # model.add(BatchNormalization())

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer='adam',
                  metrics=[fbeta_score_,fmeasure_,'acc'])

    earlystopping = EarlyStopping(monitor='val_loss',       # monitor fbeta_score_ custom function on validation data
                                  min_delta=0,
                                  patience=5,
                                  verbose=1,
                                  mode='min',
                                  restore_best_weights=True)

    output_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\"
    # output_dir = "/home/paulw/model_output/" + new_folder + "/"
    # output_dir = "/Users/Johannes/Desktop/model_output/" + new_folder + "/"
    #output_dir = "/home/johannescheck/model_output/" + new_folder + "/"

    modelcheckpoint = ModelCheckpoint(
        filepath=output_dir + 'weights-ep_{epoch:02d}-train_{fbeta_score_:.2f}-val_{val_fbeta_score_:.2f}.hdf5',
        monitor='val_fbeta_score_',
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode='max')

    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_val, y_val),
                     batch_size=60,
                     epochs=1,
                     verbose=2,
                     shuffle=True,
                     callbacks=[modelcheckpoint,tensorboard,earlystopping,metrics_])

    y_pred = np.asarray(model.predict(X_test,
                                      batch_size=60,
                                      verbose=1)).round()

    conf = confusion_matrix(y_test.argmax(axis=1),              # axis=1: from one-hot-encoding -> pred
                            y_pred.argmax(axis=1),
                            labels=y_test.argmax(axis=1))       # argmax is the inverse of to_categorical (https://github.com/keras-team/keras/issues/4981)

    # def sequence_to_text(list_of_indices):
    #     # https://stackoverflow.com/questions/41971587/how-to-convert-predicted-sequence-back-to-text-in-keras
    #     words = [reverse_word_map.get(letter) for letter in list_of_indices]
    #     print(words)
    #     return words
    #
    # print(X_test[0])
    #
    # my_texts = list(map(sequence_to_text, X_test[0]))
    # print(my_texts)
    # def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    #     """pretty print for confusion matrixes"""
    #     columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    #     empty_cell = " " * columnwidth
    #     # Print header
    #     print("    " + empty_cell, end=" ")
    #     for label in labels:
    #         print("%{0}s".format(columnwidth) % label, end=" ")
    #     print()
    #     # Print rows
    #     for i, label1 in enumerate(labels):
    #         print("    %{0}s".format(columnwidth) % label1, end=" ")
    #         for j in range(len(labels)):
    #             cell = "%{0}.1f".format(columnwidth) % cm[i, j]
    #             if hide_zeroes:
    #                 cell = cell if float(cm[i, j]) != 0 else empty_cell
    #             if hide_diagonal:
    #                 cell = cell if i != j else empty_cell
    #             if hide_threshold:
    #                 cell = cell if cm[i, j] > hide_threshold else empty_cell
    #             print(cell, end=" ")
    #         print()
    #
    # labels = [y_test.argmax(axis=1)]
    # print_cm(conf, labels)


    # try:
    #     pre, rec, f1, _ = precision_recall_fscore_support(y_test,
    #                                                       y_pred,
    #                                                       beta=1,
    #                                                       average='weighted')
    # except:
    #     pre, rec, f1 = None, None, None
    #
    # try:
    #     fb_r = fbeta_score(y_test,
    #                        y_pred,
    #                        beta=3,
    #                        average='weighted')
    #
    #     fb_p = fbeta_score(y_test,
    #                        y_pred,
    #                        beta=0.5,
    #                        average='weighted')
    # except:
    #     fb_r, fb_p = None, None
    #
    # try:
    #     eval_ = model.evaluate(X_test,
    #                            y_test,
    #                            batch_size=32)
    # except:
    #     eval_ = None
    #
    #
    # try:
    #     cr = classification_report(y_test,y_pred)
    # except:
    #     cr = None


    # print("\nTest results")
    # print("Score, FB, F1, Acc: {}" .format(eval_))
    # print("Precision: {}, Recall: {}" .format(pre,rec))
    # print("F1-Score: {}" .format(f1))
    # print("FB-Score P: {}" .format(fb_p))
    # print("FB-Score R: {}" .format(fb_r))
    # print("Classification Report : {}" .format(cr))

    # res = dict()
    # res['Score'] = eval_
    # res['Precision'] = pre
    # res['Recall'] = rec
    # res['F1'] = f1
    # res['FB_p'] = fb_p
    # res['FB_r'] = fb_r
    # res['classification report'] = cr


    # # taken from http://laid.delanover.com/debugging-a-keras-neural-network/
    # get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
    #                                   [model.layers[1].output])
    #
    # # output in test mode = 0
    # layer_output = get_3rd_layer_output([X_train, 0])[0]
    # print(layer_output)
    #
    # # output in train mode = 1
    # layer_output = get_3rd_layer_output([X_test, 1])[0]
    # print(layer_output)

    # for layer in model.layers:
    #     print("Input shape: " + str(layer.input_shape) + ". Output shape: " + str(layer.output_shape))

    # preds = model.predict_classes(X_test)
    # for i in range(len(X_test)):
    #     # print("X {}, predicted {}, true {}" .format(X_test[i], preds[i], np.argmax(y_test[i])))
    #     print("predicted {}, true {}".format(preds[i], np.argmax(y_test[i])))

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # json_ = output_dir + 'metrics.json'
    # with open(json_, 'w') as fp:
    #     json.dump(hist.history, fp, indent=4, sort_keys=True, default=str)
    #
    # json_ = output_dir + 'summary.txt'
    # with open(json_, 'w') as fp:
    #     with redirect_stdout(fp):
    #         model.summary()
    #
    #
    # json_ = output_dir + 'test_results.json'
    # with open(json_, 'w') as fp:
    #     json.dump(res, fp, indent=4, sort_keys=True, default=str)
    #
    #
    # plt.figure("Accuracy, F1-Score")
    # plt.plot(hist.history['acc'])
    # plt.plot(hist.history['val_acc'], 'r')
    # plt.plot(hist.history['fmeasure_'], 'g')
    # plt.plot(hist.history['val_fmeasure_'], 'darkorange')
    # plt.ylabel('accuracy, fmeasure_')
    # plt.xlabel('epoch')
    # plt.legend(['train_acc', 'validation_acc','train_f1','validation_f1'], loc='upper left')
    # plt.text(0.02,
    #          0.0,
    #          "Epochs {}, batch_size {}, N_Layer {}, EmbDim {}, GRUDim {}, DenDim {}, dropout {}, reg {}".format(70, 60,
    #                                                                                                 1, 128,
    #                                                                                                 128, 128,
    #                                                                                                 0.257, 0.337),
    #          fontsize=8, transform=plt.gcf().transFigure)
    # plt.subplots_adjust(bottom=0.15)
    # plt.savefig(output_dir + "acc-val_acc-f1_score-val_f1_score.png")
    #
    # plt.figure("Loss")
    # plt.plot(hist.history['loss'])
    # plt.plot(hist.history['val_loss'], 'r')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.text(0.02,
    #          0.0,
    #          "Epochs {}, batch_size {}, N_Layer {}, EmbDim {}, GRUDim {}, DenDim {}, dropout {}, reg {}".format(70, 60,
    #                                                                                                 1, 128,
    #                                                                                                 128, 128,
    #                                                                                                 0.257, 0.337),
    #          fontsize=8, transform=plt.gcf().transFigure)
    # plt.subplots_adjust(bottom=0.15)
    # plt.savefig(output_dir + "loss-val_loss.png")


    mins = (time.time() - start_) / 60
    h = (time.time() - start_) / 3600
    if h>1.0:
        print('\nPuh! Das ging jetzt ueber {} h'.format(h))
    else:
        print('\nPuh! Das ging jetzt ueber {} mins'.format(mins))

    return model