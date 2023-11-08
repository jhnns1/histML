from __future__ import print_function

from pprint import pprint
from time import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import tensorflow as tf
import seaborn as sns

import math
from matplotlib.patches import Rectangle

plt.rcParams['font.sans-serif'] = "Arial"

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
# neues Feld mit Länge des Freitexts
# df['befund'].replace(to_replace='o.n.A.',value='onA',inplace=True,regex=True)
# df['befund'].replace(to_replace=',',value='',inplace=True,regex=True)
# df['befund'].replace(to_replace=';',value='',inplace=True,regex=True)
# df['befund'].replace(to_replace='-',value='',inplace=True,regex=True)
df['length'] = df['befund'].apply(len)
# df = df[df['length']>0]

# print(df[df['label']=='9505/3'])

# print(df.head())
# print(df['length'].describe())

# Check for NaN or Zeros
# print(df.info())
# print(np.sum(df.isnull().any(axis=1)))
# print(df.isnull().any(axis=0))

# Describe the length stats
# print(df['length'].describe())
# # Längstes Freitextfeld


bg_color = '#E6E6E6'
bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

len_bylabel_mean = df.groupby('label').mean().round(2)
len_bylabel_median = df.groupby('label').median().round(2)
top3 = sorted(zip(len_bylabel_median.values,len_bylabel_median.index),reverse=True)[:3]
print(top3)
#print(len_bylabel_mean)
print(len_bylabel_median.describe())

# # ----- BOXPLOT -----
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.boxplot(len_bylabel_median.values)
#plt.axis([0, 2, 0, 550])
ax.set_facecolor(bg_color)
plt.ylabel('Mediane der Länge')


# plt.title('Mediane der Länge des Histologiebefundes pro Klasse')
# Label um -6 nach unten Schieben für richtige Höhe!
ax.annotate('8552/3',xy=(1.04,434))
ax.annotate('8571/3',xy=(1.04,99.5))
ax.annotate('9505/3',xy=(0.75,93.5))
ax.set_yticks(list(plt.yticks()[0]) + [439,100,80,31])
plt.axis([0,2,0,450])
ax.grid(color='grey',linestyle=':',axis='y')
ax.set_axisbelow(True)
plt.show()












