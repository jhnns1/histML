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

import pickle

pd.options.mode.chained_assignment = None   # copy label to label2 and assign label 'minority_class' on top if count<100; default='warn'

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
        _val_fb = fbeta_score(val_targ, val_predict, beta = 0.5, average='weighted')
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

def fbeta_score_(y_true, y_pred, beta=0.5):
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

def get_lr_metric(optimizer):
    # taken from https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

#####################################
# Model
#####################################
def load_data():
    """Load Data"""

    individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\KR_label_befund_vor_20180903.csv"
    # individual_path_to_file = "/Users/Johannes/data.csv"
    # individual_path_to_file = "/home/johannescheck/data.csv"
    # individual_path_to_file = "/home/paulw/data.csv"

    df_ = pd.read_csv(individual_path_to_file,
                      sep='$',
                      names=['label', 'befund'],
                      encoding='latin-1',
                      header=None)

    return df_


    # Load data

def data_preprocessing(val_test_size,
                       test_val_share,
                       max_len,
                       random_state=42,
                       show_basicstats=False,
                       show_excludedstats=False,
                       set_custom_n=False,
                       custom_n=20):
    df = load_data()

    print("\nData Preprocessing...")
    # To make sure that o.n.A. isn't torn apart by any tokenizer
    df['befund'].replace(to_replace='o.n.A.',
                          value='onA',
                          inplace=True,
                          regex=True)

    # Zeilenumbruch herausfiltern
    df['befund'].replace(to_replace=';',
                         value='',
                         inplace=True,
                         regex=True)

    # Zeilenumbruch herausfiltern
    df['befund'].replace(to_replace='\r\n',
                         value='',
                         inplace=True,
                         regex=True)

    # Zeilenumbruch herausfiltern
    df['befund'].replace(to_replace='\n',
                         value='',
                         inplace=True,
                         regex=True)

    if show_basicstats==True:
        print("\nShape of the dataset is: {}" .format(df.shape))
        print("Number of classes given in the dataset: {}" . format(len(np.unique(df['label']))))
        print("\nSome sanitary checks - looking for obvious hickups...")
        print(df.head())
        print(df.head(50))
        print("\nSome stats")
        print(df.describe(include='all'))
        print(df.info())

    # Only include labels with counts >= min_count_per_class to avoid train-test-split conflicts
    # And only include valid labels: >= 8000, <=9999
    df = df[df['label'] >= '8000/0']
    df = df[df['label'] < '9993/0']

    df_ = df

    df['count_label'] = df.groupby(['label']).transform('count')
    # print("\nEstablished label 'minority_class'")
    # df['label'][df['count_label'] < 6] = 'minority_class'

    # Specialized stopwords, carefully chosen regarding the task
    stopwords_ = frozenset(['und', 'bei', 'da', 'einem', 'ein', 'einer', 'eine', 'vom',
                            'dem', 'mit', 'zum', 'in', 'im', 'cm', 'mm', 'am', 'an', 'um',
                            'es', 'auf', 'für', 'als', 'aus', 'eher', 'dabei'])

    if show_excludedstats==True:
        # Calculate excluded values
        missing_labels = set(df_['label']) - set(df['label'])
        print("Labels went missing bcoz fewer than {} occurences: {}" .format(100, missing_labels))
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

    # Preprocessing, Bag-Of-Words from keras.Tokenizer()
    print('\nPreprocessing of texts...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_tok = tokenizer.texts_to_sequences(X)
    X_con = []

    for i in range(len(X_tok)):
        X_con.append('_'.join(str(e) for e in X_tok[i]))

    df['LOW'] = X_con
    df['count_befund'] = df.groupby(['befund','label','count_label']).transform('count')
    df['LOW_label'] = df['label'] + "$" + X_con
    df['count_kombi'] = df.groupby(['LOW','count_label','count_befund','label','befund']).transform('count')
    df['eindeutig'] = df['count_befund']==df['count_kombi']

    df1 = df[(df['count_befund'] > 50) & (df['eindeutig']==True)]
    df2 = df[df['count_befund'] <= 50]
    df2 = df2.append(df[(df['count_befund'] > 50) & (df['eindeutig']==False)])

    df_100 = df2[['label', 'befund']]

    df_down = df1.groupby(['label', 'befund']).size().reset_index().rename(columns={0: 'count'})
    df_down = df_down.drop('count', 1)

    for i in range(10):
        df_100 = df_100.append(df_down[['label', 'befund']])

    X = df_100['befund']
    y = to_categorical(preprocessing.LabelEncoder().fit_transform(df_100['label'].values), num_classes=None)

    # # Train-Validate-Test-Split
    print('\nSplitting the data in train, validation and test set...')
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=val_test_size,
                                                                #stratify=y,
                                                                random_state=random_state,
                                                                shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_val_share,
                                                    #stratify=y_val_test,
                                                    random_state=random_state,
                                                    shuffle=True)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    new_folder = "combined_50_{}_".format(random_state) + time.strftime("%Y%m%d_%H%M%S")
    output_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tokz = output_dir + "tokenizer.pickle"
    with open(tokz, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(tokz, 'rb') as handle:
        tokenizer = pickle.load(handle)

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

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes, output_dir

def create_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, output_dir):

    start_ = time.time()
    #log_dir = './logs_' + new_folder
    # log_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\logs_"
    # #log_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\logs_\\final\\{}".format(time())
    # #log_dir = "/home/johannescheck/model_output/" + new_folder + "/logs_"
    # # log_dir = "/Users/Johannes/Desktop/model_output/" + new_folder + "/logs_"
    # # log_dir = "/home/paulw/model_output/" + new_folder + "/logs_"
    #
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    #
    # tensorboard = TensorBoard(log_dir=log_dir,
    #                           histogram_freq=1,         # plot the distributions of weights and biases in the nn
    #                           write_graph=True,         # print the graph of nn as defined internally
    #                           write_images=True)        # create an image by combining the weight of nn

    print("\nBuilding model...")
    model = Sequential()
    model.add(Embedding(input_dim=20000,output_dim=128))
    model.add(GRU(128,
                  kernel_regularizer=regularizers.l2(0.2748),
                  activation=None))
    model.add(Activation(activation='elu'))
    model.add(Dropout(0.4465))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax"))

    # give class weights a try
    # class_weights = compute_class_weight('balanced', np.unique(np.argmax(y_train,axis=1)), np.argmax(y_train,axis=1))   # one can just use y_train bcoz of stratify = True at train-test-split
    # dict_cw = dict(enumerate(class_weights))

    adam_ = Adam(lr=0.001)
    lr_metric = get_lr_metric(adam_)
    # adadelta_ = Adadelta(clipnorm=1.0)
    model.compile(loss="categorical_crossentropy",
                  optimizer=adam_,
                  #class_weights=class_weights,
                  metrics=[fbeta_score_,fmeasure_,'acc', lr_metric])

    earlystopping = EarlyStopping(monitor='val_loss',       # monitor fbeta_score_ custom function on validation data
                                  min_delta=0,
                                  patience=5,
                                  verbose=1,
                                  mode='min',
                                  restore_best_weights=True)

    modelcheckpoint = ModelCheckpoint(
        filepath=output_dir + 'model.hdf5',
        monitor='val_fbeta_score_',
        verbose=0,
        save_best_only=True,
        mode='max')

    hist = model.fit(X_train,
                     y_train,
                     validation_data=(X_val, y_val),
                     batch_size=245,
                     epochs=60,
                     verbose=2,
                     shuffle=True,
                     # class_weight=dict_cw,
                     callbacks=[modelcheckpoint,earlystopping,metrics_])

    y_pred = np.asarray(model.predict(X_test,
                                      batch_size=245,
                                      verbose=1)).round()

    # conf = confusion_matrix(y_test.argmax(axis=1),              # axis=1: from one-hot-encoding -> pred
    #                         y_pred.argmax(axis=1),
    #                         labels=y_test.argmax(axis=1))       # argmax is the inverse of to_categorical (https://github.com/keras-team/keras/issues/4981)

    try:
        pre, rec, f1, _ = precision_recall_fscore_support(y_test,
                                                      y_pred,
                                                      beta=1,
                                                      average='weighted')
    except:
        pre, rec, f1 = None, None, None

    try:
        fb_b15 = fbeta_score(y_test,
                         y_pred,
                         beta=1.5,
                         average='weighted')

        fb_b05 = fbeta_score(y_test,
                         y_pred,
                         beta=0.5,
                         average='weighted')
    except:
        fb_b05, fb_b15 = None, None


    try:
        eval_ = model.evaluate(X_test,
                             y_test,
                             batch_size=245)
    except:
        eval_ = None

    try:
        cr = classification_report(y_test,y_pred)
    except:
        cr = None

    model_json = model.to_json()
    path_ = output_dir + "model.json"
    with open(path_, "w") as json_file:
        json_file.write(model_json)

    print("\nTest results")
    print("Score, FB, F1, Acc: {}" .format(eval_))
    print("Precision: {}, Recall: {}" .format(pre,rec))
    print("F1-Score: {}" .format(f1))
    print("FB_b05-Score: {}" .format(fb_b05))
    print("FB_b15-Score: {}" .format(fb_b15))
    print("Classification Report: {}" .format(cr))
    # print("FB_b05_nb: {}" .format(fb_b05_nb))

    # res = dict()
    # res['Score, FB_b05, F1, Acc'] = eval_
    # res['Precision'] = pre
    # res['Recall'] = rec
    # res['F1'] = f1
    # res['FB_b05'] = fb_b05
    # res['FB_b15'] = fb_b15
    # # res['FB_b05_nb'] = fb_b05_nb
    # res['classification report'] = cr
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
    # json_ = output_dir + 'test_results.json'
    # with open(json_, 'w') as fp:
    #     json.dump(res, fp, indent=4, sort_keys=True, default=str)
    #
    # plt.figure("Accuracy, FB-Measure b=0.5")
    # plt.plot(hist.history['acc'])
    # plt.plot(hist.history['val_acc'], 'r')
    # plt.plot(hist.history['fbeta_score_'], 'g')
    # plt.plot(hist.history['val_fbeta_score_'], 'darkorange')
    # plt.ylabel('accuracy, fbeta_score_')
    # plt.xlabel('epoch')
    # plt.legend(['train_acc', 'validation_acc','train_fb','validation_fb'], loc='lower right')
    # plt.savefig(output_dir + "acc-val_acc-f1_score-val_f1_score.png")
    #
    # plt.figure("Loss")
    # plt.plot(hist.history['loss'])
    # plt.plot(hist.history['val_loss'], 'r')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.savefig(output_dir + "loss-val_loss.png")

    # plt.figure("LR")
    # plt.plot(hist.history['lr'])
    # plt.plot(hist.history['val_lr'], 'r')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.savefig(output_dir + "lr-val_lr.png")


    mins = (time.time() - start_) / 60
    h = (time.time() - start_) / 3600
    if h>1.0:
        print('\nPuh! Das ging jetzt ueber {} h'.format(h))
    else:
        print('\nPuh! Das ging jetzt ueber {} mins'.format(mins))

    return model