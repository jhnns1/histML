"""
created by: @Johannes
at: 09.01.2019

Runtime for Full Model implementation w/ fb-measure, accuracy and precision as metrics

optimized by hyperopt w/ parameters for n_layer, n_neurons, L2, Activation, Dropout, BatchNorm, Batchsize, Epochs
"""

from fm_obj_v10 import create_model, data_preprocessing

X_train, y_train, X_val, y_val, X_test, y_test, num_classes = data_preprocessing(val_test_size=0.3,
                                                                                 test_val_share=0.5,
                                                                                 max_len=100,
                                                                                 random_state=42)

model = create_model(X_train, y_train, X_val, y_val, X_test, y_test,num_classes)