from __future__ import print_function

from pprint import pprint
from time import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from wordcloud import WordCloud

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

df['befund'].replace(to_replace='o.n.A.',
                     value='onA',
                     inplace=True,
                     regex=True)

mes_string = []
for t in df['befund']:
    mes_string.append(t)
mes_string = pd.Series(mes_string).str.cat(sep=' ')

# print(mes_string)
wordcloud_ = WordCloud(colormap="Reds",background_color="white",width=1600,height=800,max_font_size=200,collocations=False).generate(mes_string)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud_, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()