"""
created by: @Johannes
at: 09.01.2019

Runtime for NB, SVM implementation w/ fb-measure as metric

"""

import time
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, fbeta_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

starttime = time.time()

individual_path_to_file = "KR_label_befund_vor_20180903.csv"
# individual_path_to_file = "your_path_to_file"
individual_path_to_file2 = "KR_label_befund_nach_20180903.csv"
# individual_path_to_file = "your_path_to_file"

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
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3,
                                                            # stratify=y,
                                                            random_state=42,
                                                            shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5,
                                                # stratify=y_val_test,
                                                random_state=42,
                                                shuffle=True)


# NB, SVM Classifier erstellen
# taken from http://adataanalyst.com/scikit-learn/countvectorizer-sklearn-example/
# token pattern soll auch Wörter mit nur einem Buchstaben finden

text_clf_nb = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=(1,10))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', MultinomialNB()),])

# SGDClassifier + Hinge-Loss = Lineare SVM
text_clf_svm = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=(1,10))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=50, tol=None)),])

# Fit
text_clf_nb.fit(X_train, y_train)
text_clf_svm.fit(X_train, y_train)

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

pred_nb = text_clf_nb.predict(X_test)
pred_svm = text_clf_svm.predict(X_test)

pred_nb_new = text_clf_nb.predict(X_new)
pred_svm_new = text_clf_svm.predict(X_new)

pre, rec, _, _ = precision_recall_fscore_support(y_test,
                                              pred_nb,
                                              average='weighted')

print("\nNaive Bayes:")
print("FB_b05: {}, Accuracy: {}, Loss: {}, Precision: {}, Recall {}" .format(fbeta_score(y_test,
                  pred_nb,
                  beta=0.5,
                  average='weighted'),accuracy_score(y_test,pred_nb),0,pre,rec))

pre, rec, _, _ = precision_recall_fscore_support(y_new,pred_nb_new,average='weighted')
print("FB_b05: {}, Accuracy: {}, Loss: {}, Precision: {}, Recall: {}" .format(fbeta_score(y_new,
                  pred_nb_new,
                  beta=0.5,
                  average='weighted'),accuracy_score(y_new,pred_nb_new),0,pre,rec))

print("\nSVM:")
pre, rec, _, _ = precision_recall_fscore_support(y_test,pred_svm,average='weighted')
print("FB_b05: {}, Accuracy: {}, Loss: {}, Precision: {}, Recall: {}" .format(fbeta_score(y_test,
                  pred_svm,
                  beta=0.5,
                  average='weighted'),accuracy_score(y_test,pred_svm),0,pre,rec))
pre, rec, _, _ = precision_recall_fscore_support(y_new,pred_svm_new,average='weighted')
print("FB_b05: {}, Accuracy: {}, Loss: {}, Precision: {}, Recall: {}" .format(fbeta_score(y_new,
                  pred_svm_new,
                  beta=0.5,
                  average='weighted'),accuracy_score(y_new,pred_svm_new),0,pre,rec))

conf_label = list(set(np.unique(y_new)) | set(np.unique(pred_svm_new)))

conf = confusion_matrix(y_test,
                        pred_svm)


# taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
FP = conf.sum(axis=0) - np.diag(conf)
print([index for index,value in enumerate(FP) if value>=68])
FN = conf.sum(axis=1) - np.diag(conf)
print([index for index, value in enumerate(FN) if value >= 24])
TP = np.diag(conf)
TN = conf.sum() - (FP + FN + TP)

print("\nFP")
print("Label: {}, Count: {}" .format((conf_label[53]),FP[53]))
print("Label: {}, Count: {}" .format((conf_label[121]),FP[121]))

print("\nFN")
print("Label: {}, Count: {}" .format((conf_label[7]),FN[7]))
print("Label: {}, Count: {}" .format((conf_label[53]),FN[53]))

# print(len(TP))
# print(len(set(np.unique(y_test.argmax(axis=1))) | set(np.unique(y_pred.argmax(axis=1)))))
print("TP: {}" .format(TP))
print(TP.shape)
print(np.sum(TP))
print("FP: {}".format(FP))
print(FP.shape)
print(np.sum(FP))
print("FN: {}".format(FN))
print(np.sum(FN))

conf = confusion_matrix(y_new,
                        pred_svm_new)


# taken from https://stackoverflow.com/questions/50666091/compute-true-and-false-positive-rate-tpr-fpr-for-multi-class-data
FP = conf.sum(axis=0) - np.diag(conf)
print([index for index,value in enumerate(FP) if value>=90])
FN = conf.sum(axis=1) - np.diag(conf)
print([index for index, value in enumerate(FN) if value >= 70])
TP = np.diag(conf)
TN = conf.sum() - (FP + FN + TP)

print("\nFP")
print("Label: {}, Count: {}" .format((conf_label[58]),FP[58]))
print("Label: {}, Count: {}" .format((conf_label[132]),FP[132]))

print("\nFN")
print("Label: {}, Count: {}" .format((conf_label[9]),FN[9]))
print("Label: {}, Count: {}" .format((conf_label[54]),FN[54]))
print("Label: {}, Count: {}" .format((conf_label[58]),FN[58]))

# print(len(TP))
# print(len(set(np.unique(y_test.argmax(axis=1))) | set(np.unique(y_pred.argmax(axis=1)))))
print("TP: {}" .format(TP))
print(TP.shape)
print(np.sum(TP))
print("FP: {}".format(FP))
print(FP.shape)
print(np.sum(FP))
print("FN: {}".format(FN))
print(np.sum(FN))


