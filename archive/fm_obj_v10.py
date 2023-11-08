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
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, classification_report, confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sn

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
                       show_excludedstats=False):
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

    X = df['befund'].values
    print('\nLabel-Encoding...')
    # entweder use to_categorical + categorical_crossentropy
    # oder use just preprocessing + sparse_categorical_crossentropy

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df['label'].values)
    reversed_label = dict(zip(y, df['label'].values))
    y = to_categorical(y, num_classes=None)

    # store the current # of classes for use in output layer
    num_classes = y.shape[1]

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

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(X)

    new_folder = "fm_obj_v10_{}_".format(random_state) + time.strftime("%Y%m%d_%H%M%S")
    output_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # saving tokenizer
    # tokz = output_dir + "tokenizer.pickle"
    # with open(tokz, 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # loading saved tokenizer
    tokz = "C:\\Users\\johannes.heck\\Desktop\\model_output\\fm_v10\\tokenizer.pickle"
    with open(tokz, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # saving reversed label
    revlabel = "C:\\Users\\johannes.heck\\Desktop\\model_output\\fm_v10\\reversed_label.pickle"
    with open(revlabel, 'wb') as handle:
        pickle.dump(reversed_label, handle, pickle.HIGHEST_PROTOCOL)

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
    model.add(Embedding(input_dim=20000,output_dim=256))
    model.add(GRU(256,
                  kernel_regularizer=regularizers.l2(0.0295),
                  activation=None,
                  return_sequences=True))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.5978))
    model.add(BatchNormalization())
    model.add(GRU(256,
                  kernel_regularizer=regularizers.l2(0.0295),
                  activation=None))
    model.add(Activation(activation='relu'))
    model.add(Dropout(0.2639))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax"))

    print("\nLoaded model from fm_v10...")
    model.load_weights("C:\\Users\\johannes.heck\\Desktop\\model_output\\fm_v10\\model.hdf5")

    adam_ = Adam(lr=0.001)
    lr_metric = get_lr_metric(adam_)
    # adadelta_ = Adadelta(clipnorm=1.0)

    print("\nCompiling...")
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

    # hist = model.fit(X_train,
    #                  y_train,
    #                  validation_data=(X_val, y_val),
    #                  batch_size=210,
    #                  epochs=65,
    #                  verbose=2,
    #                  shuffle=True,
    #                  # class_weight=dict_cw,
    #                  callbacks=[modelcheckpoint,earlystopping,metrics_])

    print("\nPredicting...")
    y_pred = np.asarray(model.predict(X_test,
                                      batch_size=210,
                                      verbose=1)).round()

    # loading saved reversed label
    revlabel = "C:\\Users\\johannes.heck\\Desktop\\model_output\\fm_v10\\reversed_label.pickle"
    with open(revlabel, 'rb') as handle:
        reversed_label = pickle.load(handle)

    sorted_reversed_label = sorted(reversed_label.items(), key = lambda x: x[1])
    sorted_label = [x[1] for x in sorted_reversed_label]

    # loading saved tokenizer
    tokz = "C:\\Users\\johannes.heck\\Desktop\\model_output\\fm_v10\\tokenizer.pickle"
    with open(tokz, 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("\nEvaluating...")
    # try:
    #     conf = confusion_matrix(y_test.argmax(axis=1),              # axis=1: from one-hot-encoding -> pred
    #                             y_pred.argmax(axis=1))       # argmax is the inverse of to_categorical (https://github.com/keras-team/keras/issues/4981)
    # except:
    #     conf = None
    conf_label = list(set(np.unique(y_test.argmax(axis=1))) | set(np.unique(y_pred.argmax(axis=1))))

    conf = confusion_matrix(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1))


    # taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
    FP = conf.sum(axis=0) - np.diag(conf)
    print([index for index,value in enumerate(FP) if value>=43])
    FN = conf.sum(axis=1) - np.diag(conf)
    print([index for index, value in enumerate(FN) if value >= 24])
    TP = np.diag(conf)
    TN = conf.sum() - (FP + FN + TP)

    print("\nFP")
    print("Label: {}, Count: {}" .format(reversed_label.get(conf_label[0]),FP[0]))
    print("Label: {}, Count: {}" .format(reversed_label.get(conf_label[54]),FP[54]))
    print("Label: {}, Count: {}" .format(reversed_label.get(conf_label[122]),FP[122]))

    print("\nFN")
    print("Label: {}, Count: {}" .format(reversed_label.get(conf_label[7]),FN[7]))
    print("Label: {}, Count: {}" .format(reversed_label.get(conf_label[54]),FN[54]))
    print("Label: {}, Count: {}" .format(reversed_label.get(conf_label[131]),FN[131]))

    # print(len(TP))
    # print(len(set(np.unique(y_test.argmax(axis=1))) | set(np.unique(y_pred.argmax(axis=1)))))
    print("TP: {}" .format(TP))
    print(TP.shape)
    print(np.sum(TP))
    print("FP: {}".format(FP))
    print(FP.shape)
    print(np.sum(FP))
    print("FN: {}".format(FN))
    print(np.sum(FN))

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


    # try:
    #     eval_ = model.evaluate(X_test,
    #                          y_test,
    #                          batch_size=210)
    # except:
    #     eval_ = None

    try:
        cr = classification_report(y_test,y_pred,target_names=sorted_label)
    except:
        cr = None


    individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\KR_label_befund_nach_20180903.csv"
    # individual_path_to_file = "/Users/Johannes/data.csv"
    # individual_path_to_file = "/home/johannescheck/data.csv"
    # individual_path_to_file = "/home/paulw/data.csv"

    df_new = pd.read_csv(individual_path_to_file,
                      sep='$',
                      names=['label', 'befund'],
                      encoding='latin-1',
                      header=None)

    df_new['befund'].replace(to_replace='o.n.A.',
                         value='onA',
                         inplace=True,
                         regex=True)

    # Zeilenumbruch herausfiltern
    df_new['befund'].replace(to_replace=';',
                         value='',
                         inplace=True,
                         regex=True)

    # Zeilenumbruch herausfiltern
    df_new['befund'].replace(to_replace='\r\n',
                         value='',
                         inplace=True,
                         regex=True)

    # Zeilenumbruch herausfiltern
    df_new['befund'].replace(to_replace='\n',
                         value='',
                         inplace=True,
                         regex=True)

    df_new = df_new[df_new['label'] >= '8000/0']
    df_new = df_new[df_new['label'] < '9993/0']

    X_new = df_new['befund'].values
    X_tok = tokenizer.texts_to_sequences(X_new)
    X_new = sequence.pad_sequences(X_tok,100)

    y_pred_label = (np.asarray(model.predict(X_new,
                                      batch_size=210,
                                      verbose=1)).round()).argmax(axis=1)
    tmp = y_pred_label
    y_pred_label = [reversed_label.get(i) for i in y_pred_label]
    y_label = df_new['label']

    conf_new = confusion_matrix(y_label,
                                y_pred_label)

    conf_label = list(set(np.unique(y_label)) | set(np.unique(y_pred_label)))

    # taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
    FP = conf_new.sum(axis=0) - np.diag(conf_new)
    print([index for index,value in enumerate(FP) if value>=79])
    FN = conf_new.sum(axis=1) - np.diag(conf_new)
    print([index for index, value in enumerate(FN) if value >= 53])
    TP = np.diag(conf_new)
    TN = conf_new.sum() - (FP + FN + TP)

    print("\nFP")
    print("Label: {}, Count: {}" .format(conf_label[0],FP[0]))
    print("Label: {}, Count: {}" .format((conf_label[56]),FP[56]))
    print("Label: {}, Count: {}" .format((conf_label[132]),FP[132]))

    print("\nFN")
    print("Label: {}, Count: {}" .format((conf_label[9]),FN[9]))
    print("Label: {}, Count: {}" .format((conf_label[29]),FN[29]))
    print("Label: {}, Count: {}" .format((conf_label[48]),FN[48]))
    print("Label: {}, Count: {}".format((conf_label[56]), FN[56]))
    print("Label: {}, Count: {}".format((conf_label[166]), FN[166]))

    # print(len(TP))
    # print(len(set(np.unique(y_test.argmax(axis=1))) | set(np.unique(y_pred.argmax(axis=1)))))
    print("TP: {}" .format(TP))
    print(TP.shape)
    print(np.sum(TP))
    print("FP: {}".format(FP))
    print(FP.shape)
    print(np.sum(FP))
    print("FN: {}".format(FN))
    print(np.sum(FN))

    print(np.reshape(y_pred_label, (-1, 1)))
    print(y_label)

    y_pred_df = pd.DataFrame(np.reshape(y_pred_label, (-1,1)))
    print(y_pred_df)

    try:
        cr_label = classification_report(y_label,y_pred_label)
    except:
        cr_label = None

    try:
        pre, rec, f1, _ = precision_recall_fscore_support(y_label,
                                                      y_pred_df,
                                                      beta=0.5,
                                                      average='weighted')
    except:
        pre, rec, f1 = None, None, None

    try:
        fb_b15 = fbeta_score(y_label,
                         y_pred_df,
                         beta=1.5,
                         average='weighted')

        fb_b05 = fbeta_score(y_label,
                         y_pred_df,
                         beta=0.5,
                         average='weighted')
    except:
        fb_b05, fb_b15 = None, None


    try:
        eval_ = model.evaluate(X_new,
                             y_label,
                             batch_size=210)
    except:
        eval_ = None

    print("\nFinal results")
    print("Score, FB, F1, Acc: {}" .format(eval_))
    print("Precision: {}, Recall: {}" .format(pre,rec))
    print("F1-Score: {}" .format(f1))
    print("FB_b05-Score: {}" .format(fb_b05))
    print("FB_b15-Score: {}" .format(fb_b15))
    print("Classification Report: {}" .format(cr_label))
    #
    # res = dict()
    # res['Score, FB_b05, F1, Acc'] = eval_
    # res['Precision'] = pre
    # res['Recall'] = rec
    # res['F1'] = f1
    # res['FB_b05'] = fb_b05
    # res['FB_b15'] = fb_b15
    # res['classification report'] = cr
    # res['confusion matrix'] = conf
    #
    # json_ = output_dir + 'summary.txt'
    # with open(json_, 'w') as fp:
    #     with redirect_stdout(fp):
    #         model.summary()
    #
    # json_ = output_dir + 'test_results.json'
    # with open(json_, 'w') as fp:
    #     json.dump(res, fp, indent=4, sort_keys=True, default=str)


    mins = (time.time() - start_) / 60
    h = (time.time() - start_) / 3600
    if h>1.0:
        print('\nPuh! Das ging jetzt ueber {} h'.format(h))
    else:
        print('\nPuh! Das ging jetzt ueber {} mins'.format(mins))

    return model