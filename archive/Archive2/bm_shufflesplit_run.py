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
from bm_shufflesplit_obj import config_

min_count_per_class, max_features, max_len, batch_size, epochs, val_test_size, test_val_share, lr_, dropout_, reg_ = config_(batch_size_=32, epochs_=100, lr_=0.001, dropout_=0.2, reg_=0.01)

from bm_shufflesplit_obj import create_model, data_preprocessing

X_train, y_train, X_test, y_test, num_classes = data_preprocessing(0.3,
                                                                   80)

model = create_model(X_train, y_train, X_test, y_test,num_classes)


