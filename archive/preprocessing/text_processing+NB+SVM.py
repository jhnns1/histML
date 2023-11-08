import sklearn
import pandas as pd
import numpy as np
from glob import glob
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, fbeta_score, classification_report, confusion_matrix, roc_auc_score, log_loss, precision_recall_fscore_support

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

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

df1 = df[df['count_befund'] > 250]
df2 = df[df['count_befund'] <= 250]

all_words1 = df1['label'].value_counts()
all_words2 = df2['label'].value_counts()

# X = df2['befund'].values
# y = df2['label'].values

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
#X_train, X_test, y_train, y_test = train_test_split(messages_['morph_freitext'], messages_['label'], test_size=0.3)
# X_train, X_validate_test, y_train, y_validate_test = train_test_split(messages_['morph_freitext'], messages_['label'], test_size=0.0, random_state=SEED, stratify=messages_['label'])
# X_validate, X_test, y_validate, y_test = train_test_split(X_validate_test, y_validate_test, test_size=0.5, random_state=SEED, stratify=y_validate_test)

# taken from http://adataanalyst.com/scikit-learn/countvectorizer-sklearn-example/


text_clf_nb = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=(1,10))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', MultinomialNB()),])

# include words with just one letter -> token_pattern =
# o.n.A. zu einem Word deklarieren?! Howto?!
# linear support vector machine SVM
text_clf_svm = Pipeline([('vect', CountVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b", ngram_range=(1,10))),
                     ('tfidf', TfidfTransformer(use_idf=True)),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=50, tol=None)),])

text_clf_nb.fit(X_train, y_train)
text_clf_svm.fit(X_train, y_train)

individual_path_to_file = "C:\\Users\\johannes.heck\\Desktop\\Data\\KR_label_befund_nach_20180903.csv"
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

le = preprocessing.LabelEncoder()

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


