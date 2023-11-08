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
from keras.preprocessing.text import Tokenizer

import logging

import plotly.offline as py
import plotly.graph_objs as go
from sklearn.preprocessing import LabelEncoder
plt.rcParams['font.sans-serif'] = "Arial"

# Establish logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\KR_label_befund_vor_20180903.csv"
# individual_path_to_file = "/Users/Johannes/data.csv"
# individual_path_to_file = "/home/johannescheck/data.csv"
# individual_path_to_file = "/home/paulw/data.csv"

df = pd.read_csv(individual_path_to_file,
                  sep='$',
                  names=['label', 'befund'],
                  encoding='latin-1',
                  header=None)

df['befund'].replace(to_replace='o.n.A.',
                     value='onA',
                     inplace=True,
                     regex=True)

# # Zeilenumbruch herausfiltern
# df['befund'].replace(to_replace=';',
#                      value='',
#                      inplace=True,
#                      regex=True)
#
# # Zeilenumbruch herausfiltern
# df['befund'].replace(to_replace='\r\n',
#                      value='',
#                      inplace=True,
#                      regex=True)

bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

df = df[df['label'] >= '8000/0']
df = df[df['label'] < '9993/0']

df_ = df

df['count_label'] = df.groupby(['label']).transform('count')

X = df['befund'].values

print('\nPreprocessing of texts...')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_tok = tokenizer.texts_to_sequences(X)
X_con = []

for i in range(len(X_tok)):
    X_con.append('_'.join(str(e) for e in X_tok[i]))

df['LOW'] = X_con
# print(df.groupby(['LOW', 'label', 'count_label']).transform('count'))
df['count_befund'] = df.groupby(['befund', 'label', 'count_label']).transform('count')

df['LOW_label'] = df['label'] + "$" + X_con
df['count_kombi'] = df.groupby(['LOW', 'count_label', 'count_befund', 'label', 'befund']).transform('count')

df['eindeutig'] = df['count_befund'] == df['count_kombi']

df1 = df[df['count_label'] >= 0]
df1 = df1[df1['count_label'] < 5]
print(len(df1))


# df1 = df[df['count_befund'] > 250]
# print(len(df1))
# df2 = df[df['count_befund'] <= 250]
#
all_words = df['label'].value_counts()
# print(all_words.describe())
#
# all_words1 = df1['label'].value_counts()
# all_words2 = df2['label'].value_counts()
#
# tmp = np.unique(df2['befund'][df2['label']=='8140/3'])
# print(len(tmp))
#
# # tmp.to_csv("C:\\Users\\johannes.heck\\Desktop\\df_tmp.csv", sep="~", encoding='latin1')
#
#
fig, ax = plt.subplots()
fig.set_figheight(15)
n = 30
#ax.yaxis.grid(True)
ax.grid(color='grey',linestyle=':',axis='x')
plt.barh(all_words.index.values[0:n], all_words.values[0:n],facecolor=fg_color,alpha=1,edgecolor=bg_color)
# plt.xticks(rotation=90)
plt.xlabel('Häufigkeit')
plt.ylabel('ICD-O-M Codes')
xt = ax.get_xticks()
xt = xt[1:]
xt = np.append(xt,[500])
ax.set_xticks(xt)
#plt.title('Die {} häufigsten Histologiecodes' .format(n))
plt.subplots_adjust(left=0.15)
#plt.xlim(-0.5,)
plt.axis([0, 16000, -1, n])
ax.invert_yaxis()
ax.set_axisbelow(True)
ax.set_facecolor(bg_color)
plt.show()