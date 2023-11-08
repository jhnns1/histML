"""
created by: @Johannes
at: 09.01.2019

Runtime for NB, SVM implementation w/ fb-measure, accuracy and precision as metrics

optimized by GridSearchCV w/ parameters for ngram_range, alpha, tf_idf-use
"""

import warnings
warnings.filterwarnings('ignore') # filter UndefinedMetricWarning for F1, FBeta for Classes that are set to 0

import time
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

start_ = time.time()

individual_path_to_file = "KR_label_befund_vor_20180903.csv"
individual_path_to_file2 = "KR_label_befund_nach_20180903.csv"

print("Loading & Cleaning Data...")
df = pd.read_csv(individual_path_to_file,
                  sep='$',
                  names=['label', 'befund'],
                  encoding='latin-1',
                  header=None)

# "o.n.A." zu einem Ausdruck machen
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

# inkonsistente Label herausfiltern, vgl. DIMDI
df = df[df['label'] >= '8000/0']
df = df[df['label'] < '9993/0']

X = df['befund'].values
y = df['label'].values

print('\nSplitting the data in train, validation and test set...')
X_train, X_val_test, y_train, y_val_test = train_test_split(X,
                                                            y,
                                                            test_size=0.3,
                                                            # stratify=y,
                                                            random_state=42,
                                                            shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_val_test,
                                                y_val_test,
                                                test_size=0.5,
                                                # stratify=y_val_test,
                                                random_state=42,
                                                shuffle=True)

# NB, SVM Classifier erstellen
# taken from http://adataanalyst.com/scikit-learn/countvectorizer-sklearn-example/
# token pattern soll auch Wörter mit nur einem Buchstaben finden

print("\nCreating Classifier...")
text_clf_nb = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=(1,10))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', MultinomialNB()),])

# SGDClassifier + Hinge-Loss = Lineare SVM
text_clf_svm = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=(1,10))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=50, tol=None)),])

print("\nFitting models...")
text_clf_nb.fit(X_train, y_train)
text_clf_svm.fit(X_train, y_train)

print("\nLoading & Cleaning new Data...")
df = pd.read_csv(individual_path_to_file2,
                  sep='$',
                  names=['label', 'befund'],
                  encoding='latin-1',
                  header=None)

# "o.n.A." zu einem Ausdruck machen
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

# inkonsistente Label herausfiltern, vgl. DIMDI
df = df[df['label'] >= '8000/0']
df = df[df['label'] < '9993/0']

X_new = df['befund'].values
y_new = df['label'].values

print("\nPredicting...")
pred_nb = text_clf_nb.predict(X_test)
pred_svm = text_clf_svm.predict(X_test)

pred_nb_new = text_clf_nb.predict(X_new)
pred_svm_new = text_clf_svm.predict(X_new)

print("\nEvaluation Naive Bayes...")
print("\n...on Test Data...")
print("\nFB_b05: {}, \nAccuracy: {}, \nPrecision: {}, " .format(fbeta_score(y_test,
                                                          pred_nb,
                                                          beta=0.5,
                                                          average='weighted'),
                                                          accuracy_score(y_test,
                                                                         pred_nb),
                                                          precision_score(y_test,
                                                                          pred_nb,
                                                                          average='weighted')))
conf = confusion_matrix(y_test,
                        pred_nb)
# taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
FP = conf.sum(axis=0) - np.diag(conf)
FN = conf.sum(axis=1) - np.diag(conf)
TP = np.diag(conf)
TN = conf.sum() - (FP + FN + TP)

print("TP: {}" .format(np.sum(TP)))
print("FP: {}".format(np.sum(FP)))
print("FN: {}".format(np.sum(FN)))

print("\n...on New Data...")
print("\nFB_b05: {}, \nAccuracy: {}, \nPrecision: {}" .format(fbeta_score(y_new,
                                                                          pred_nb_new,
                                                                          beta=0.5,
                                                                          average='weighted'),
                                                              accuracy_score(y_new,
                                                                             pred_nb_new),
                                                              precision_score(y_new,
                                                                              pred_nb_new,
                                                                              average='weighted')))

conf = confusion_matrix(y_new,
                        pred_nb_new)

# taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
FP = conf.sum(axis=0) - np.diag(conf)
FN = conf.sum(axis=1) - np.diag(conf)
TP = np.diag(conf)
TN = conf.sum() - (FP + FN + TP)

print("TP: {}" .format(np.sum(TP)))
print("FP: {}".format(np.sum(FP)))
print("FN: {}".format(np.sum(FN)))

print("\nEvaluation SVM...")
print("\n...on Test Data...")
print("\nFB_b05: {}, \nAccuracy: \nPrecision: {}" .format(fbeta_score(y_test,
                                                                      pred_svm,
                                                                      beta=0.5,
                                                                      average='weighted'),
                                                          accuracy_score(y_test,
                                                                         pred_svm),
                                                          precision_score(y_test,
                                                                          pred_svm,
                                                                          average='weighted')))
conf = confusion_matrix(y_test,
                        pred_svm)
# taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
FP = conf.sum(axis=0) - np.diag(conf)
FN = conf.sum(axis=1) - np.diag(conf)
TP = np.diag(conf)
TN = conf.sum() - (FP + FN + TP)

print("TP: {}" .format(np.sum(TP)))
print("FP: {}".format(np.sum(FP)))
print("FN: {}".format(np.sum(FN)))

print("\n...on New Data...")
print("\nFB_b05: {}, \nAccuracy: {}, \nPrecision: {}" .format(fbeta_score(y_new,
                                                                          pred_svm_new,
                                                                          beta=0.5,
                                                                          average='weighted'),
                                                              accuracy_score(y_new,
                                                                             pred_svm_new),
                                                              precision_score(y_new,
                                                                              pred_svm_new,
                                                                              average='weighted')))
conf = confusion_matrix(y_new,
                        pred_svm_new)

# taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
FP = conf.sum(axis=0) - np.diag(conf)
FN = conf.sum(axis=1) - np.diag(conf)
TP = np.diag(conf)
TN = conf.sum() - (FP + FN + TP)

print("TP: {}" .format(np.sum(TP)))
print("FP: {}".format(np.sum(FP)))
print("FN: {}".format(np.sum(FN)))

mins = (time.time() - start_) / 60
h = (time.time() - start_) / 3600
if h > 1.0:
    print('\nPuh! Das ging jetzt ueber {} h'.format(h))
else:
    print('\nPuh! Das ging jetzt ueber {} mins'.format(mins))