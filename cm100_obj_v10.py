"""
created by: @Johannes
at: 09.01.2019

Objective for Combined Model w/ rules threshold 100 implementation w/ fb-measure, accuracy and precision as metrics

optimized by hyperopt w/ parameters for n_layer, n_neurons, L2, Activation, Dropout, BatchNorm, Batchsize, Epochs

Current implementation: Load trained model, tokenizer and label-dict
For Training: uncomment model.fit part in create_model, disable loading trained model
"""

import warnings
warnings.filterwarnings('ignore') # filter UndefinedMetricWarning for F1, FBeta for Classes that are set to 0

import numpy as np
import pandas as pd
import time
import os

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import GRU
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, classification_report, confusion_matrix, accuracy_score

import os.path

import pickle

pd.options.mode.chained_assignment = None   # copy label to label2 and assign label 'minority_class' on top if count<100; default='warn'

if os.name == 'nt':
    sep_ = "\\"
else:
    sep_ = "/"

###################################################################
# Customized Metrics that are not available in keras 2.0 anymore
###################################################################

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

###################################################################
# Model and Routines
###################################################################

def load_data():
    individual_path_to_file = "KR_label_befund_vor_20180903.csv"

    df = pd.read_csv(individual_path_to_file,
                      sep='$',
                      names=['label', 'befund'],
                      encoding='latin-1',
                      header=None)

    return df

def data_preprocessing(val_test_size,
                       test_val_share,
                       max_len,
                       random_state=42):
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

    # Only include valid labels: >= 8000, <=9999
    df = df[df['label'] >= '8000/0']
    df = df[df['label'] < '9993/0']

    df['count_label'] = df.groupby(['label']).transform('count')

    X = df['befund'].values

    # Preprocessing, Bag-Of-Words from keras.Tokenizer()
    print('\nTokenizing // Text to sequences...')
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    X_tok = tokenizer.texts_to_sequences(X)
    X_con = []

    # hier werden Regeln mit Threshold 100 erstell, indem das Label mit der Sequenz von Einträgen kombiniert wird
    # (= LOW_label)
    # und anschliessend auf Eindeutigkeit geprüft wird
    # df2 beinhaltet dann alle Befund-Label-Kombinationen, die seltener als 100 mal vorkommen oder nicht eindeutig sind
    # anschliessend findet eine Wiedereinsteuerung der Kombinationen mit dem Median von df2 statt, um Rechtschreibfehler
    # abzufangen

    print('\nEstablishing rules w/ unique label-befund Combo...')
    for i in range(len(X_tok)):
        X_con.append('_'.join(str(e) for e in X_tok[i]))

    df['LOW'] = X_con
    df['count_befund'] = df.groupby(['befund','label','count_label']).transform('count')
    # füge Label + konkatenierten Befund zusammen
    df['LOW_label'] = df['label'] + "$" + X_con
    df['count_kombi'] = df.groupby(['LOW','count_label','count_befund','label','befund']).transform('count')
    df['eindeutig'] = df['count_befund']==df['count_kombi']

    df1 = df[(df['count_befund'] > 100) & (df['eindeutig']==True)]
    df2 = df[df['count_befund'] <= 100]
    df2 = df2.append(df[(df['count_befund'] > 100) & (df['eindeutig']==False)])

    df_100 = df2[['label', 'befund']]

    print("\nExporting rules...")
    # df_down enthält alle "Regeln" (Label-Befund-Kombination, die eindeutig sind und mehr als 100 Mal vorkommen)
    df_down = df1.groupby(['label', 'befund']).size().reset_index().rename(columns={0: 'count'})
    df_down.to_pickle("cm100_v10" + sep_ + "rules100.pkl")
    df_down = df_down.drop('count', 1)

    # Wiedereinsteuerung der Regel-Datensätze, um Rechtschreibfehler abzufangen; verzerrt Metriken nur minimal
    for i in range(10):
        df_100 = df_100.append(df_down[['label', 'befund']])

    print("\nApplicating rules...")
    X = df_100['befund']

    print('\nLabel-Encoding...')
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df_100['label'].values)
    reversed_label = dict(zip(y, df_100['label'].values))
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

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)

    # tokz = "tokenizer.pickle"
    # with open(tokz, 'wb') as handle:
    #     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # loading saved tokenizer
    tokz = "cm100_v10" + sep_ + "tokenizer.pickle"
    with open(tokz, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # saving reversed label
    revlabel = "cm100_v10" + sep_ + "reversed_label.pickle"
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

    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes

def create_model(X_train, y_train, X_val, y_val, X_test, y_test, num_classes):

    start_ = time.time()

    print("\nBuilding model...")
    model = Sequential()
    model.add(Embedding(input_dim=20000,output_dim=64))
    model.add(GRU(64,
                  kernel_regularizer=regularizers.l2(0.2115),
                  activation=None))
    model.add(Activation(activation='tanh'))
    model.add(Dropout(0.2836))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax"))

    print("\nLoaded model from cm100_v10...")
    model.load_weights("cm100_v10" + sep_ + "model.hdf5")

    print("\nCompiling Model...")
    model.compile(loss="categorical_crossentropy",
                  optimizer=Adam(lr=0.001),
                  metrics=[fbeta_score_,fmeasure_,'acc'])

    earlystopping = EarlyStopping(monitor='val_loss',       # monitor fbeta_score_ custom function on validation data
                                  min_delta=0,
                                  patience=5,
                                  verbose=1,
                                  mode='min',
                                  restore_best_weights=True)

    modelcheckpoint = ModelCheckpoint(
        filepath='model.hdf5',
        monitor='val_fbeta_score_',
        verbose=0,
        save_best_only=True,
        mode='max')

    # hist = model.fit(X_train,
    #                  y_train,
    #                  validation_data=(X_val, y_val),
    #                  batch_size=195,
    #                  epochs=110,
    #                  verbose=2,
    #                  shuffle=True,
    #                  # class_weight=dict_cw,
    #                  callbacks=[modelcheckpoint,earlystopping,metrics_])

    print("\nPredicting Test Data...")
    y_pred = np.asarray(model.predict(X_test,
                                      batch_size=195,
                                      verbose=0)).round()

    # loading saved reversed label
    revlabel = "cm100_v10" + sep_ + "reversed_label.pickle"
    with open(revlabel, 'rb') as handle:
        reversed_label = pickle.load(handle)

    # sorted_label is used for actual labels in classification_report
    sorted_reversed_label = sorted(reversed_label.items(), key=lambda x: x[1])
    sorted_label = [x[1] for x in sorted_reversed_label]

    # loading saved tokenizer
    tokz = "cm100_v10" + sep_ + "tokenizer.pickle"
    with open(tokz, 'rb') as handle:
        tokenizer = pickle.load(handle)

    print("\nEvaluating Test Data...")

    try:
        pre, rec, f1, _ = precision_recall_fscore_support(y_test,
                                                      y_pred,
                                                      beta=1,
                                                      average='weighted')
    except:
        pre, rec, f1 = None, None, None

    try:
        fb_b05 = fbeta_score(y_test,
                         y_pred,
                         beta=0.5,
                         average='weighted')
    except:
        fb_b05 = None


    try:
        eval_ = model.evaluate(X_test,
                               y_test,
                               batch_size=195,
                               verbose=0)
    except:
        eval_ = None

    try:
        cr = classification_report(y_test,y_pred,sorted_label)
    except:
        cr = None

    print("\nTest results")
    # print("Score, F1, FB, Acc: {}" .format(eval_))
    print("FBeta (b=0.5): {}" .format(fb_b05))
    print("Precision: {}" .format(pre))
    # print("Recall: {}" .format(rec)) # Recall-Wert = Accuracy, fehlerhaft; daher per Hand berechnet
    # print("Classification Report: {}" .format(cr))

    conf = confusion_matrix(y_test.argmax(axis=1),
                            y_pred.argmax(axis=1))

    # Berechne TP, FP, FN, TN
    # taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
    FP = conf.sum(axis=0) - np.diag(conf)
    FN = conf.sum(axis=1) - np.diag(conf)
    TP = np.diag(conf)
    TN = conf.sum() - (FP + FN + TP)

    print("TP: {}".format(np.sum(TP)))
    print("FP: {}".format(np.sum(FP)))
    print("FN: {}".format(np.sum(FN)))

    print("\nLoading & Preprocessing New Data...")
    individual_path_to_file = "KR_label_befund_nach_20180903.csv"

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

    df_rules = pd.read_pickle("cm100_v10" + sep_ + "rules100.pkl")
    X_rules = df_rules['befund'].values

    # filter X_rules and remove "~" all befund that are in X_rules
    df_new = df_new[~df_new['befund'].isin(X_rules)]

    X_new = df_new['befund'].values
    X_tok = tokenizer.texts_to_sequences(X_new)
    X_new = sequence.pad_sequences(X_tok, 100)

    print("\nPredicting new Data...")
    y_pred_label = (np.asarray(model.predict(X_new,
                                             batch_size=210,
                                             verbose=0)).round()).argmax(axis=1)

    y_pred_label = [reversed_label.get(i) for i in y_pred_label]
    y_label = df_new['label']
    y_pred_df = pd.DataFrame(np.reshape(y_pred_label, (-1,1)))

    print("\nEvaluating new Data...")
    try:
        pre, rec, f1, _ = precision_recall_fscore_support(y_label,
                                                      y_pred_df,
                                                      beta=0.5,
                                                      average='weighted')
    except:
        pre, rec, f1 = None, None, None

    try:
        fb_b05 = fbeta_score(y_label,
                         y_pred_df,
                         beta=0.5,
                         average='weighted')
    except:
        fb_b05 = None

    # try:
    #     eval_ = model.evaluate(X_new,
    #                          y_label,
    #                          batch_size=210,
    #                           verbose=0)
    # except:
    #     eval_ = None
    
    # try:
    #     cr_label = classification_report(y_label,y_pred_label)
    # except:
    #     cr_label = None

    print("\nFinal results")
    # print("Score, F1, FB, Acc: {}" .format(eval_))
    print("FB: {}" .format(fb_b05))
    print("Precision: {}" .format(pre))
    # print("Recall: {}" .format(rec)) # Recall-Wert = Accuracy, fehlerhaft; daher per Hand berechnet
    # print("Classification Report: {}" .format(cr_label))

    conf_new = confusion_matrix(y_label,
                                y_pred_label)

    # Berechne TP, FP, FN, TN
    # taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
    FP = conf_new.sum(axis=0) - np.diag(conf_new)
    FN = conf_new.sum(axis=1) - np.diag(conf_new)
    TP = np.diag(conf_new)
    TN = conf_new.sum() - (FP + FN + TP)

    print("TP: {}".format(np.sum(TP)))
    print("FP: {}".format(np.sum(FP)))
    print("FN: {}".format(np.sum(FN)))

    mins = (time.time() - start_) / 60
    h = (time.time() - start_) / 3600
    if h>1.0:
        print('\nPuh! Das ging jetzt ueber {} h'.format(h))
    else:
        print('\nPuh! Das ging jetzt ueber {} mins'.format(mins))

    return model