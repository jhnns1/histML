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

import plotly.offline as py
import plotly.graph_objs as go

from matplotlib.patches import Rectangle

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
df = df[df['label'] >= '8000/0']
df = df[df['label'] < '9993/0']

# print(df.head(50))

df['befund_einzeln'] = df['befund'].str.split(expand=False)#.unstack().value_counts()
# bg_color = 'white'
# fg_color = '#00868b'
# fg_color2 = '#EC7563'
#
# fig, ax = plt.subplots()
# #ax.yaxis.grid(True)
# plt.bar(all_words.index.values[0:30], all_words.values[0:30],facecolor=fg_color,alpha=1,edgecolor=bg_color)
# plt.xticks(rotation=90)
# #plt.xlabel('Medizinische Fachbegriffe')
# plt.ylabel('Häufigkeit')
# plt.title('TOP 30 medizinische Fachbegriffe')
# plt.subplots_adjust(bottom=0.4)
# #plt.xlim(-0.5,)
# plt.axis([-1, 30, 0, 32000])
# ax.set_facecolor(bg_color)
# plt.show()
df['anzahl_woerter_befund'] = df['befund_einzeln'].apply(len)
# print(df['anzahl_woerter_befund'])
# print(df['anzahl_woerter_befund'].describe())
# print(df['anzahl_woerter_befund'].describe())
# shorties = df[df['anzahl_woerter_befund']<2]
# shorties['len_'] = shorties['befund'].apply(len)
# shorties = shorties.drop(['anzahl_woerter_befund','befund_einzeln'],axis=1)
# exp_ = shorties[shorties['len_']<=6]
# exp_.to_csv("C:\\Users\\johannes.heck\\Desktop\\shorties.csv", sep="~", encoding='latin1')

bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'


# # # ----- BOXPLOT -----
# fig, ax = plt.subplots(figsize=(5,5))
# plt.boxplot(df['length'])
# plt.axis([0, 2, 0, 550])
# ax.set_facecolor(bg_color)
# plt.ylabel('Länge des Freitextfeldes')
# plt.title('Histogramm der Länge des Histologiebefundes')
# # ax.yaxis.grid(True)
# #plt.grid(True)
# plt.show()

# ----- HISTO -----
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.grid(color='grey',linestyle=':',axis='y')
# n, bins, patches = plt.hist(df['anzahl_woerter_befund'],bins=400,density=True,facecolor=fg_color,alpha=1,edgecolor=bg_color)
# plt.axvline(df['anzahl_woerter_befund'].median(), color='k', linestyle='--', linewidth=1)
# plt.axvline(df['anzahl_woerter_befund'].mean(), color='k', linestyle='-.', linewidth=1)
# plt.text(df['anzahl_woerter_befund'].mean()+0.3,0.9,round(df['anzahl_woerter_befund'].mean(),2))
# plt.axis([0, 10, 0, 1.5])
# ax.set_facecolor(bg_color)


plt.boxplot(df['anzahl_woerter_befund'].values)
ax.set_yticks(list(plt.yticks()[0][2:]) + [111,3])
plt.ylabel('Anzahl Fachbegriffe')


# plt.xlabel('Anzahl Fachbegriffe')
# plt.ylabel('Häufigkeit')
# extra = Rectangle((0,0), 1, 1, fc='w', fill=False, edgecolor='None', linewidth=0)
# ax.set_xticks(list(plt.xticks()[0][:]) + [1,3,5,7,9])
# plt.legend(["Median","Mittelwert"], handlelength=3)
# plt.title('Histogramm der Länge des Histologiebefundes')
# ax.yaxis.grid(True,linestyle=':',alpha=0.75)
#plt.grid(False)
ax.set_axisbelow(True)
#plt.show()