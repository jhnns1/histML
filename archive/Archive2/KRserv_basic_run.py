# -*- coding: utf-8 -*-

import numpy as np
np.random.seed(13) # for reproducibility

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from hyperopt import fmin, tpe, hp, Trials
from hyperopt.mongoexp import MongoTrials

import matplotlib.pyplot as plt

import logging
import json
import os, time
import pickle

# Customizing
# Use AVX2 (for windows): https://github.com/fo40225/tensorflow-windows-wheel, install via pip install <filename>
# Mac: https://github.com/lakshayg/tensorflow-build

plt.rcParams['font.sans-serif'] = "Arial"
bg_color = 'white'
fg_color = '#00868b'
fg_color2 = '#EC7563'

# Establish logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




# Hyperopt
# taken from https://github.com/keras-team/keras/issues/1591 and
# https://github.com/Vooban/Hyperopt-Keras-CNN-CIFAR-100

# space = {   'dropout': hp.uniform('dropout', .1,.6),
#             'batch_size' : hp.quniform('batch_size',16,512,5),
#             'n_layer' : hp.choice('n_layer', [1,2,3,4,5]),
#             'n_neuron' : hp.choice('n_neuron', [32, 64, 128, 256, 512, 1024]),
#             'optimizer': hp.choice('optimizer',['Adam','RMSprop','Nadam','AdaDelta']),
#             'activation': hp.choice('activation',['relu','tanh','sigmoid','elu','selu']),
#             'use_BN' : hp.choice('use_BN', [False,True]),
#             'L2reg': hp.loguniform('L2reg', -1.3,1.3)
#         }

space = {   'dropout1': hp.uniform('dropout1', .1,.6),
            'dropout2': hp.uniform('dropout2', .1,.6),
            'batch_size' : hp.quniform('batch_size',16,256,5),
            'n_layer' : hp.choice('n_layer', [1,2,3]),
            'n_neuron1' : hp.choice('n_neuron1', [16, 32, 128, 256, 512, 1024]),
            'n_neuron2': hp.choice('n_neuron2', [16, 32, 128, 256, 512, 1024]),
            #'LR' : hp.choice('LR', 0.01, 0.00001),
            'activation': hp.choice('activation',['relu','tanh','elu']),
            'use_BN' : hp.choice('use_BN', [False,True]),
            'L2reg': hp.uniform('L2reg', 0.01,0.4),
            'epochs': hp.quniform('epochs',50,200,5)
        }


# inspiration
#trials = Trials()

from KRserv_basic_obj import create_model

# trials = Trials()
# trials = pickle.load(open("myfile.p", "rb"))
trials = MongoTrials('mongo://localhost:1234/foo_db/jobs', exp_key='exp40')
best = fmin(create_model,               # 'Loss' function to minimize
            space,                      # Hyperparameter space
            algo=tpe.suggest,           # Tree-structured Parzen Estimator (TPE)
            max_evals=15,
            trials=trials)

print('\nBest model is:')
try:
    print("\nWith max FBeta b=0.5: ")
    print(abs(min(trials.losses())))
except:
    pass


new_folder = time.strftime("%Y%m%d_%H%M%S")
#output_dir = "C:\\Users\\johannes.heck\\Desktop\\model_output\\" + new_folder + "\\"
output_dir = "/home/paulw/model_output/" + new_folder + "/"
#output_dir = "/home/johannescheck/model_output/" + new_folder + "/"
# output_dir = "/Users/Johannes/Desktop/model_output/" + new_folder + "/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    trials_stored = output_dir + "trials_stored.p"
    pickle.dump(trials, open(trials_stored, "wb"))
except:
    pass

json_ = output_dir + 'best.json'
with open(json_, 'w') as fp:
    json.dump(best, fp, indent=4, sort_keys=True, default=str)

json_ = output_dir + 'trials_trials.json'
with open(json_,'w') as fp:
    json.dump(trials.trials, fp, indent=4, sort_keys=True, default=str)

json_ = output_dir + 'trials_results.json'
with open(json_,'w') as fp:
    json.dump(trials.results, fp, indent=4, sort_keys=True, default=str)

json_ = output_dir + 'trials_losses.json'
with open(json_,'w') as fp:
    json.dump(trials.losses(), fp, indent=4, sort_keys=True, default=str)

json_ = output_dir + 'trials_statuses.json'
with open(json_,'w') as fp:
    json.dump(trials.statuses(), fp, indent=4, sort_keys=True, default=str)