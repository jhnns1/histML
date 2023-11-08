# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(13) # for reproducibility

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from hyperopt import fmin, tpe, hp, Trials

import matplotlib.pyplot as plt

import logging
import json
import os, time

# Customizing
# Use AVX2 (for windows): https://github.com/fo40225/tensorflow-windows-wheel, install via pip install <filename>
# Mac: https://github.com/lakshayg/tensorflow-build

plt.rcParams['font.sans-serif'] = "Arial"
bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

# Establish logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Parameters
from combined_50_obj import create_model, data_preprocessing

rs = [42, 0, 13, 5, 99]
for state in rs:
    X_train, y_train, X_val, y_val, X_test, y_test, num_classes, output_dir = data_preprocessing(0.3,
                                                                                     0.5,
                                                                                     100,
                                                                                     random_state=state)

    model = create_model(X_train, y_train, X_val, y_val, X_test, y_test,num_classes,output_dir)


