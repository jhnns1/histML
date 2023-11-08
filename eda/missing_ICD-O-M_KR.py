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

import logging

from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = "Arial"

# Establish logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

df_m = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\Data\\Archiv\\20180921-icdo_m_all.csv', sep=';',names=['label','morph_freitext'],encoding='latin-1')
df_10_c = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\Data\\Archiv\\20180921-icd10_icdo.csv', sep=';',names=['ICD-10','ICD-O-C'],encoding='latin-1')
df_kr = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\Data\\Archiv\\20180921-label+feature_all.csv', sep=';',names=['label','morph_freitext','ICD-10','ICD-O-C'],encoding='latin-1')

# # # -------------- ICD-O M missing ------------------
# missing_m = list(set(df_m['label']) - set(df_kr['label']))
# print(missing_m)
# print(len(missing_m)-1)

print(df_kr['ICD-O-C'].head(50))

icd10 = df_kr['ICD-O-C'].value_counts()
icd10_per = (df_kr['ICD-O-C'].value_counts() / len(df_kr)).sort_values(ascending=False)
print(icd10_per)