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

df = pd.read_csv('C:\\Users\\johannes.heck\\Desktop\\20180921-label+feature_all.csv', sep=';',names=['label','morph_freitext','ICD10','ICDOC'],encoding='latin-1')
# print("Shape of dataset is: {}" .format(df.shape))
# print(df.head())
# print(df.head(50))
# all_words = df['label'].value_counts()
all_words = df['ICD10'].value_counts()#
all_words2 = df['ICDOC'].value_counts()
bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

print(all_words.index.values[:30])
print(all_words2.index.values[:30])

all_words.index.values[0] = 'C61 / C61.9'
all_words2.index.values[0] = 'C61 / C61.9'

all_words.index.values[4] = 'C20 / C20.9'
all_words2.index.values[4] = 'C20 / C20.9'

# all_words.index.values[1] = 'C44.3 / C44.{31,32,33,34,53}'
# all_words2.index.values[5] = 'C44.3 / C44.{31,32,33,34,53}'
# all_words2.index.values[14] = 'C44.3 / C44.{31,32,33,34,53}'
# all_words2.index.values[18] = 'C44.3 / C44.{31,32,33,34,53}'
# all_words2.index.values[19] = 'C44.3 / C44.{31,32,33,34,53}'
# all_words2.index.values[20] = 'C44.3 / C44.{31,32,33,34,53}'
# all_words2.index.values[28] = 'C44.3 / C44.{31,32,33,34,53}'

fig, ax = plt.subplots()
#ax.yaxis.grid(True)
plt.bar(all_words.index.values[0:30], all_words.values[0:30],facecolor=fg_color,alpha=1,edgecolor=bg_color)
plt.bar(all_words2.index.values[0:30], all_words2.values[0:30], facecolor=fg_color2,alpha=.75, edgecolor=bg_color)
plt.xticks(rotation=90)
plt.title('TOP 30 ICD-10 gegen TOP 30 ICD-O C')
plt.ylabel('Häufigkeit des Vorkommens')
plt.subplots_adjust(bottom=0.25)
fontP = FontProperties()
fontP.set_size('small')
plt.legend(["ICD-10","ICD-O C"], handlelength=3, prop=fontP)
#plt.xlim(-0.5,)
plt.axis([-1, 38, 0, 5000])
ax.set_facecolor(bg_color)
plt.show()