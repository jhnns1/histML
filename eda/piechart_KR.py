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

from matplotlib import cm

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

import logging

import plotly.offline as py
import plotly.graph_objs as go
import seaborn as sns

from matplotlib import rc

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

# print("Shape of dataset is: {}" .format(df.shape))
# print(df.head())
# print(df.info())
df['text'] = np.zeros(len(df))
# print(df['label'].value_counts())

# # ----------- PIE CHART ------------
percent_ = (df[['label','text']].groupby('label').count()/len(df)).sort_values(by='text', ascending=False)
df = percent_.reset_index()

label = df['label'][:10]
label[10] = 'Rest'
sizes = df['text'][:10]
sizes[10] = round(1-sum(sizes),3)

fig1, ax1 = plt.subplots(figsize=(6.5,6))
cmaps = (sns.diverging_palette(220, 20, n=11))
#cmaps = ['salmon', 'darkred', 'turquoise', 'teal', 'orange', 'brown', 'coral', 'lightblue', 'lavender', 'tan', 'grey']
ax1.pie(sizes, colors=cmaps, labels=label, autopct='%1.1f%%', startangle=90)
centre_circle=plt.Circle((0,0), 0.7, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# plt.title("Anteile der Klassen an der Gesamtheit")
ax1.axis('equal')
plt.tight_layout()
plt.show()





