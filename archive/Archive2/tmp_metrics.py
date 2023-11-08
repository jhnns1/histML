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
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

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

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from gensim.models import Word2Vec

import logging
import time
from sys import platform

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

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys

# Parameters
from bm_global_metrics import set_hyper

min_count_per_class, max_features, max_len, batch_size, epochs, val_test_size, test_val_share, lr_, dropout_, reg_ = set_hyper(batch_size_=32, epochs_=100, lr_=0.001, dropout_=0.2, reg_=0.01)

from bm_global_metrics import data_preprocessing

X_train, y_train, X_val, y_val, X_test, y_test, num_classes = data_preprocessing(min_count_per_class, val_test_size, test_val_share, max_len, set_custom_n=True, custom_n=20)
#X_train, y_train, X_val, y_val, X_test, y_test, num_classes = data_preprocessing(min_count_per_class, val_test_size, test_val_share, max_len)

from bm_global_metrics import create_model
model = create_model(X_train, y_train, X_val, y_val, X_test, y_test, batch_size, epochs, max_features, embed_dim=256, gru_dim=256, den_dim=num_classes, dropout_=dropout_, lr_=lr_, reg_=reg_)

