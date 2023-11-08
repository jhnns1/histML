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

# print(df.head())
print(df['length'].describe())

# Check for NaN or Zeros
# print(df.info())
# print(np.sum(df.isnull().any(axis=1)))
# print(df.isnull().any(axis=0))

# Describe the length stats
# print(df['length'].describe())
# # Längstes Freitextfeld
print(df[df['length'] == 929]['befund'].iloc[0])

#bg_color = '#E6E6E6'
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
n, bins, patches = plt.hist(df['length'],bins=600,density=True,facecolor=fg_color,alpha=1,edgecolor=bg_color)
plt.axvline(df['length'].median(), color='k', linestyle='--', linewidth=1)
plt.text(df['length'].median()-4,0.05,round(df['length'].median(),2))
plt.axvline(df['length'].mean(), color='k', linestyle='-.', linewidth=1)
plt.text(df['length'].mean()+1,0.06,round(df['length'].mean(),2))
plt.axis([0, 60, 0, 0.135])
ax.set_facecolor(bg_color)
plt.xlabel('Anzahl Zeichen')
plt.ylabel('Häufigkeit')
extra = Rectangle((0,0), 1, 1, fc='w', fill=False, edgecolor='None', linewidth=0)
plt.legend(["Median","Mittelwert"], handlelength=3)
plt.title('Histogramm der Länge des Histologiebefundes')
ax.yaxis.grid(True,linestyle=':',alpha=0.75)

# plt.boxplot(df['length'].values)
# plt.ylabel('Anzahl Zeichen')
# ax.grid(color='grey',linestyle=':',axis='y')
# ax.set_yticks(list(plt.yticks()[0][2:]) + [929,80,31])


ax.set_axisbelow(True)
plt.show()

# # ----- by label -----
# avg_len = df['length'].mean()
# avg_len_bylabel = df.groupby('label').mean().round(2)
# print(avg_len_bylabel)
#
# print(avg_len_bylabel.describe())












