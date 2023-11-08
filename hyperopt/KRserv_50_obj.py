# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)


import numpy as np
import pandas as pd
import sys
import math

pd.options.mode.chained_assignment = None   # default='warn'

import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.layers import GRU, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import Callback
from keras.optimizers import Adam, Adadelta, Nadam

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold, ShuffleSplit
from sklearn.utils import compute_class_weight

from imblearn.over_sampling import SMOTE

from hyperopt import STATUS_OK



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

    val_test_size = 0.4
    test_val_share = 0.5

    lr = lr_
    dropout = dropout_
    reg = reg_
    return min_count_per_class, max_features, max_len, batch_size, epochs, val_test_size, test_val_share, lr, dropout, reg

# Parameters

min_count_per_class, max_features, max_len, batch_size, epochs, val_test_size, test_val_share, lr_, dropout_, reg_ = config_(batch_size_=32, epochs_=100, lr_=0.001, dropout_=0.2, reg_=0.01)

def load_data(individual_path_to_file=""):
    """Load Data"""

    #individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\20180905-morph_code+morph_freitext.csv"
    # individual_path_to_file = "/Users/Johannes/data.csv"
    # individual_path_to_file = "/home/johannescheck/data.csv"
    individual_path_to_file = "/home/paulw/data.csv"

    df_ = pd.read_csv(individual_path_to_file,
                      sep='$',
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
    # Kodierungsfehler werden idR durch >= min_count_per_class herausgefiltert
    df = df[df['label'] >= '8000/0']
    df = df[df['label'] < '9993/0']
    df_ = df
    # And only include valid labels: >= 8000, <=9999

    df['count_label'] = df.groupby(['label']).transform('count')
    df['len_befund'] = df['befund'].apply(len)

    print(len(df))
    df = df[df['len_befund'] <= max_len]
    print(len(df))
    df = df.drop('len_befund', 1)
    # print("\nEstablished label 'minority_class'")
    # df['label'][df['count_label'] < 6] = 'minority_class'
    # print(len(df[df['label']=='minority_class']))

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

    # # Preprocessing, Bag-Of-Words from keras.Tokenizer()
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
    df2 = df2.append(df[(df['count_befund'] > 50) & (df['eindeutig'] == False)])

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
                                                                random_state=42,
                                                                shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_val_share,
                                                    #stratify=y_val_test,
                                                    random_state=42,
                                                    shuffle=True)

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

X_train, y_train, X_val, y_val, X_test, y_test, num_classes = data_preprocessing(6,
                                                                                 0.3,
                                                                                 0.5,
                                                                                 100)

def create_model(params):
    print("\nParams testing: ", params)

    print("\nBuilding model...")
    model = Sequential()
    model.add(Embedding(input_dim=20000,output_dim=params['n_neuron1']))

    tmp = params['n_layer']
    while tmp >= 2:
        model.add(GRU(params['n_neuron1'],
                      kernel_regularizer=regularizers.l2(params['L2reg']),
                      return_sequences=True,
                      # dropout=params['dropout1'],
                      # recurrent_dropout=params['dropout2'],
                      # activation=params['activation']))
                      activation=None))
        model.add(Activation(params['activation']))
        model.add(Dropout(params['dropout2']))
        if params['use_BN']==True:
            model.add(BatchNormalization())
        tmp -= 1

    model.add(GRU(params['n_neuron1'],
                  kernel_regularizer=regularizers.l2(params['L2reg']),
                  # dropout=params['dropout1'],
                  # recurrent_dropout=params['dropout2'],
                  # activation=params['activation']))
                  activation=None))
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))
    if params['use_BN']==True:
        model.add(BatchNormalization())
    model.add(Dense(num_classes, activation="softmax"))

    # give class weights a try
    # class_weights = compute_class_weight('balanced', np.unique(np.argmax(y_train,axis=1)), np.argmax(y_train,axis=1))   # one can just use y_train bcoz of stratify = True at train-test-split
    # dict_cw = dict(enumerate(class_weights))

    # Norm clipping
    #taken from https: // github.com / keras - team / keras / issues / 1370
    # norm = math.sqrt(sum(np.sum(K.get_value(w)) for w in model.optimizer.weights))

    adam_ = Adam(lr=0.001)
    adadelta_ = Adadelta(clipnorm=1.)
    nadam_ = Nadam(clipnorm=1.)

    model.compile(loss="categorical_crossentropy",
                  optimizer=adam_,
                  metrics=[fbeta_score_,fmeasure_,'acc'])

    earlystopping = EarlyStopping(monitor='val_loss',       # monitor fbeta_score_ custom function on validation data
                                  min_delta=0,
                                  patience=5,
                                  verbose=0,
                                  mode='min',
                                  restore_best_weights=True)

    hist = model.fit(X_train,
                          y_train,
                          validation_data=(X_val,y_val),
                          batch_size=int(params['batch_size']),
                          epochs=int(params['batch_size']),
                          verbose=2,
                          shuffle=True,
                          # class_weight=dict_cw,
                          callbacks=[earlystopping,metrics_])

    y_pred = np.asarray(model.predict(X_test,
                                      batch_size=int(params['batch_size']),
                                      verbose=1)).round()

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
        fb_b15, fb_b05 = None, None


    try:
        eval_ = model.evaluate(X_test,
                               y_test,
                               batch_size=int(params['batch_size']))
    except:
        eval_ = None


    try:
        cr = classification_report(y_test,y_pred)
    except:
        cr = None


    sys.stdout.flush()
    return {'loss' : -fb_b05,                                   # hyperopt and fmin will search for the min, so take neg.# test loss from model.evaluate
            'score' : eval_[0],                             # test loss from model.evaluate
            'status': STATUS_OK,
            'f1_test' : f1,
            'pre_test' : pre,
            'rec_test' : rec,
            'accuracy_test' : eval_[3],
            'fb_b05_test' : fb_b05,
            'fb_b15_test' : fb_b15,
            'f1_test aus eval_' : eval_[2],
            'fb_test aus eval_' : eval_[1],
            'classification report_test': cr,

            'fbeta_val' : metrics_.val_fbs,                 # as many values as epochs (bcoz applied at end of epoch)
            'f1_val' : metrics_.val_f1s,
            'precision_val' : metrics_.val_precisions,
            'recall_val' : metrics_.val_recalls,

            'params' : params}