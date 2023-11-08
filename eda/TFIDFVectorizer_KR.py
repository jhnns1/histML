import sklearn
import pandas as pd
import numpy as np
from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn import metrics
import matplotlib.pyplot as plt
import collections

import time

starttime = time.time()

individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\KR_label_befund_vor_20180903.csv"
# individual_path_to_file = "/Users/Johannes/data.csv"
# individual_path_to_file = "/home/johannescheck/data.csv"
# individual_path_to_file = "/home/paulw/data.csv"

df = pd.read_csv(individual_path_to_file,
                  sep='$',
                  names=['label', 'befund'],
                  encoding='latin-1',
                  header=None)

# neues Feld mit Länge des Freitexts
df['length'] = df['befund'].apply(len)
df['befund'].replace(to_replace='o.n.A.',value='onA',inplace=True,regex=True)

# Vorkommen der Label auf >= 100 beschränken
counts = df['label'].value_counts()
stopwords_ = frozenset(['und','bei','da','einem','ein','einer','eine','vom','dem','mit','zum','in','im','cm','mm','am','an','um','es','auf','für','als','aus','eher','dabei','übrigen','übrige','üblicherweise','übersicht','übersandten','übersand','zumindest','zumeist','zu'])

# Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(df['befund'], df['label'], test_size=0.0)
# X_train, X_validate_test, y_train, y_validate_test = train_test_split(df_['befund'], df_['label'], test_size=0.0, random_state=SEED, stratify=df_['label'])
# X_validate, X_test, y_validate, y_test = train_test_split(X_validate_test, y_validate_test, test_size=0.5, random_state=SEED, stratify=y_validate_test)

tf = TfidfVectorizer(analyzer='word',token_pattern=r"(?u)\b\w+\b",ngram_range=(1,2))
tfidf_matrix = tf.fit_transform(X_train)
feature_names = tf.get_feature_names()

df_tfidf = []
for doc in range(len(X_train)):
    feature_index = tfidf_matrix[doc,:].nonzero()[1]
    tfidf_scores = zip(feature_index, [tfidf_matrix[doc,x] for x in feature_index])

    for w, s in [(feature_names[i], s) for (i,s) in tfidf_scores]:
        df_tfidf.append([w,s])

word = [item[0] for item in df_tfidf]
tfidf = [item[1] for item in df_tfidf]

df = pd.DataFrame({'wort':word, 'tfidf':tfidf})
df = df.groupby(df['wort']).median()

df = df.sort_values(by=['tfidf'],ascending=False)

df.to_csv("C:\\Users\\johannes.heck\\Desktop\\tfidf_words.csv", sep="~", encoding='latin1')