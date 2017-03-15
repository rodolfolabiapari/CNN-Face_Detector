#!/usr/bin/python

# Generating the backend

# Search files on path
import os.path
#
import sys

import cPickle

# Backend
from neon.backends import gen_backend

from neon.data import ArrayIterator
from neon.layers import Conv, Affine, Pooling, GeneralizedCost
from neon.initializers import Uniform
from neon.transforms.activation import Rectlin, Softmax
from neon.transforms import CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.callbacks.callbacks import Callbacks

# download the image
import urllib
# crop and resize to 32x32
from PIL import Image
import numpy as np
from neon.initializers import Gaussian

import time

import basic_functs
import cnn_functs

print "[INFO]: Library read."

print "[SETU]: Setting the CPU."

# TODO alterar o batch para o valor de execuoces
be = gen_backend(backend="cpu", batch_size=1)

print "[INFO]: Information about backend:"

# Testing for verification
print "\t", be

# TODO arrumar o args
train_again = basic_functs.verify_args()
train_again = "n"

print "[DOWN]: Downloading the dataset of training."

"""
if data_set == "cifar_10":
    print "[INFO]: Loading CIFAR_10"

    # The datas are encapsulated in dataset class objetcs which contain all the
    #    metadata requerid for accessing the data.
    cifar10_dataset = CIFAR10(path="./data_sets/")

    # We separate the sets of training and test from metadata
    train_set = cifar10_dataset.train_iter
    test_set = cifar10_dataset.valid_iter

    number_class = 10

elif data_set == "cifar_100":
    print "[INFO]: Loading CIFAR_100"

    labels_people_train = []
    labels_people_test = []

    train_dictionary = unpickle(directory="./data_sets/cifar-100-python/train")
    test_dictionary = unpickle(directory="./data_sets/cifar-100-python/test")

    print "[INFO]: Package's Informations"

    print "\tDictionary:    ", train_dictionary.keys()
    print "\tData: Lines:   ", len(train_dictionary['data'])
    print "\t\t  Columns: ", len(train_dictionary['data'][0])
    print "\tCoarse: Lines: ", len(train_dictionary['coarse_labels']), "\tNumber of classes:    ", max(train_dictionary['coarse_labels'])
    print "\tFine: Lines:   ", len(train_dictionary['fine_labels']), "\tNumber of subclasses: ", max(train_dictionary['fine_labels'])

    number_class = max(train_dictionary['coarse_labels']) + 1



    print "[INFO]: Organizing the files to the training and test"

    start = time.time()

    # Train Section

    for i in range(0, len(train_dictionary['coarse_labels'])):

        if train_dictionary['coarse_labels'][i] == num_classe_dictionary:
            labels_people_train.append(1)
        else:
            labels_people_train.append(0)

    print "\tSize of train: ", len(labels_people_train)
    print "\tSum photos People Train: ", sum(labels_people_train)

    train_np = np.array(train_dictionary['data'])
    labels_people_train_np = np.array(labels_people_train)

    # train_set = ArrayIterator(X=train_np, y=labels_people_train_np, nclass=2, lshape=(3, 32, 32))
    train_set = ArrayIterator(X=train_np, y=np.array(train_dictionary['coarse_labels']),
                              nclass=number_class, lshape=(3, 32, 32))

    # Test Section

    for i in range(0, len(test_dictionary['coarse_labels'])):

        if test_dictionary['coarse_labels'][i] == num_classe_dictionary:
            labels_people_test.append(1)
        else:
            labels_people_test.append(0)

    print "\tSize of test: ", len(labels_people_test)
    print "\tSum photos People Test: ", sum(labels_people_test)

    test_np = np.array(test_dictionary['data'])
    labels_people_test_np = np.array(labels_people_test)

    # test_set = ArrayIterator(X=test_np, y=labels_people_test_np, nclass=2, lshape=(3, 32, 32))
    test_set = ArrayIterator(X=test_np, y=np.array(test_dictionary['coarse_labels']),
                              nclass=number_class, lshape=(3, 32, 32))

    end = time.time()

    print "[INFO]: Time spend to organize: ", end - start, 'seconds'

    print "[INFO]: Loading CIFAR_100 done"

"""
# TODO folders
train_set = cnn_functs.loading_set_for_training("./data_sets/FDDB-folds/FDDB-fold-01-ellipseList-1.txt")


if train_again == "y":
    print "[INFO]: The Neural Network was not created yet."
    print "[INFO]: Starting procedure of creating and training."

    # Cost

    print "[INFO]: Creating the cost function."
    # Setting up the cost function of network output
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())

    # Optimizer

    print "[INFO]: Setting the optimizer with values: learning_rate=0.005, momentum_coef=0.9."
    # Optimizer
    #    Having the cost function, we want minimize it.
    optimizer = GradientDescentMomentum(learning_rate=0.005, momentum_coef=0.9)
    # optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

    # Build Model

    model = cnn_functs.build_model(16, 128)

    # Callback

    print "[INFO]: Creating the Callbacks."
    callbacks = Callbacks(model, train_set)

    # Training

    print "[INFO]: Making the Neural Network with \"5\" epochs."
    print "\tEpoch: ;\tBatches: Quantity Images to train;\tCost: ;"

    start = time.time()
    model.fit(dataset=train_set, cost=cost, optimizer=optimizer, num_epochs=5, callbacks=callbacks)
    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    # Saving

    print "[BACK]: Saving the model with the name \"cnn-trained_model.prm\"."
    model.save_params("cnn-trained_model.prm")

else:
    print "[INFO]: Network already created."
    print "[LOAD]: Loading the model with the name \"cnn-trained_model.prm\"."

    model = Model("cnn-trained_model.prm")


# Todo Folders
test_set = cnn_functs.loading_set_for_testing("./data_sets/FDDB-folds/FDDB-fold-01-1.txt")


miss_test = False# or True
error_pct = 0
print "[INFO]: Checking the Misclassification of error."
start = time.time()
if miss_test:
    error_pct = 100 * model.eval(train_set, metric=Misclassification())
end = time.time()
print "\tTime spend to organize: ", end - start, 'seconds'
print "\tMiss classification error = %.3f%%" % error_pct
