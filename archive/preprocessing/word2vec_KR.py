import sklearn
import pandas as pd
import numpy as np
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
from collections import OrderedDict
from textblob import TextBlob
import math

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning, module='gensim')

import gzip
import gensim
from gensim.models import FastText
import logging
import time

from numpy import array

# full working example, taken from
# http://kavita-ganesan.com/gensim-word2vec-tutorial-starter-code/#.W5i-F_ZCSHt

# Establish logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

starttime = time.time()
SEED = 2000
shuffle_ = True
# messages = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\20180912-morph_freitext.csv', sep=';',names=['label','morph_freitext'],encoding='latin-1')
#
# # o.n.A. als ein Wort konservieren
# messages['morph_freitext'].replace(to_replace='o.n.A.',value='onA',inplace=True,regex=True)
#
# # Vorkommen der Label auf >= 10 beschränken
# counts = messages['label'].value_counts()
# messages_ = messages[messages['label'].isin(counts[counts >= 10].index)]

# input_file = 'C:\\Users\\johannes.heck\\Desktop\\20180912-morph_freitext.csv'
input_file = 'C:\\Users\\johannes.heck\\Desktop\\20180912-morph_freitext_all4word2vec.csv'

# with open(input_file, 'rt', encoding="latin1") as f:
#     for i, line in enumerate(f):
#         print(line)
#         if ((i+1) % 10000) == 0:
#             break


# # read files into a list
def read_input(input_file):
    """This method reads the input file which is in csv format"""
    logging.info("reading file {0}...das könnte eine Weile dauern." .format(input_file))
    with open(input_file, 'rt', encoding="latin1") as f:
        for i, line in enumerate(f):
            if (i%10000 == 0):
                logging.info("read {0} Freitexte" .format(i))
            yield gensim.utils.simple_preprocess(line,min_len=1,max_len=50)

# add list to avoid error: https://stackoverflow.com/questions/34166369/generator-is-not-an-iterator

# ------ ACHTUNG -----
# simple_preprocess trennt o.n.A., entweder csv vorbehandeln oder Ausnahme einfügen, wohl nicht per simple_preprocess möglich
messages = list(read_input(input_file))


# # build vocabulary and train model, sg=1 for skipgram
model_sg = gensim.models.Word2Vec(messages,
                               size=100,
                               window=5,
                               min_count=0,
                               workers=10,
                               sg=1)
model_sg.train(messages, total_examples=len(messages), epochs=10)

# # Sanity checks
# print(len(model.wv.vocab))
# print(model.wv.vocab)

w1 = "karzinom"
# print(model_sg.wv.most_similar(positive=w1,topn=10))
# print(model_sg.wv.similarity(w1="karzinom",w2="adenokarzinom"))
# print(model_sg.wv.similarity(w1="karzinom",w2="karzinom"))

model_ft = FastText(messages,
                    size=100,
                    window=5,
                    min_count=5,
                    workers=4,
                    sg=1)

# Erkenntnis: FastText sucht eher nach Rechtschreibung ähnlich, SG wohl eher von Sinn her; Check mit Dokumentar
print(model_ft.wv.most_similar(w1,topn=10))
print(model_sg.wv.most_similar(w1,topn=10))




