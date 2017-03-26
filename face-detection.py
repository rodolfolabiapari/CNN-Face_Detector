#!/usr/bin/python

from settings import *

from neon.backends import gen_backend
import os.path
import sys
from neon.transforms import Misclassification
from neon.models import Model

import urllib                    # download the np_image
from PIL import Image            # crop and resize to 32x32
import numpy as np               # Work with vectors and matrices
import basic_functs
# Search files on path
import os.path
import cnn_functs
import basic_functs
#
import sys
import os
import shutil

from neon.data import CIFAR10, ArrayIterator
from random import randrange
import numpy as np
import sys
import math
from neon.data import ArrayIterator
from neon.layers import Conv, Affine, Pooling, GeneralizedCost
from neon.initializers import Uniform, Gaussian
from neon.transforms.activation import Rectlin, Softmax
from neon.transforms import CrossEntropyMulti, CrossEntropyBinary, Misclassification
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.callbacks.callbacks import Callbacks
# Search files on path
import os.path
#
import sys
import time
import cPickle

# crop and resize to 32x32
from PIL import Image
import numpy as np
from random import randrange
from neon.data import ArrayIterator


def main():

    start_program = time.time()

    # Start
    print "[INFO]: Library read."
    print "[SETU]: Setting the CPU."

    # Starting the procedures of configurations

    backend = gen_backend(backend="cpu", batch_size=CONST_batch_size)
    print "[INFO]: Batch_size:", CONST_batch_size, "\tSize of the Images:", CONST_size_image_algorithm

    print "[INFO]: Information about backend:"
    print "\t", backend, "\n"

    # TODO folders from terminal reading from file
    l_directories_fddb_train = [
         "./data_sets/FDDB-folds/FDDB-fold-01-ellipseList.txt",
         "./data_sets/FDDB-folds/FDDB-fold-02-ellipseList.txt",
         "./data_sets/FDDB-folds/FDDB-fold-03-ellipseList.txt",
         "./data_sets/FDDB-folds/FDDB-fold-04-ellipseList.txt",
         "./data_sets/FDDB-folds/FDDB-fold-05-ellipseList.txt"
    ]

    # TODO read files
    diretorios_test = ["./data_sets/FDDB-folds/FDDB-fold-06.txt"
                       ]

    # Load the images creating the training set
    if CONST_train_again == "y":
        train_set = cnn_functs.loading_set_for_training(num_files_94, l_directories_fddb_train, CONST_size_image_algorithm)

    test_set, figure_test_set = cnn_functs.loading_set_for_testing(diretorios_test, CONST_batch_size, CONST_size_image_algorithm)

    # Verify if there is necessity of re-train
    if CONST_train_again == "y":

        model = cnn_functs.train_model_fddb_94(train_set, test_set, CONST_num_epochs, CONST_learning_rate, CONST_momentum)

        # Saving

        print "[BACK]: Saving the model with the name \"cnn-trained_model.prm\".\n\n"
        model.save_params(
            "./cnn/cnn-trained_model-" + str(num_files_94) + "-" + str(len(l_directories_fddb_train)) + "-" +
            str(CONST_size_image_algorithm) + "-" +
            str(CONST_num_epochs) + ".prm")

    else:
        print "[INFO]: Network already created."
        print "[LOAD]: Loading the model with the name \"cnn-trained_model.prm\".\n\n"

        if os.path.exists("./cnn/cnn-trained_model-" + str(num_files_94) + "-" + str(len(l_directories_fddb_train)) + "-" + str(CONST_size_image_algorithm) + "-" + str(CONST_num_epochs) + ".prm"):

            model = Model("./cnn/cnn-trained_model-" + str(num_files_94) + "-" + str(len(l_directories_fddb_train)) + "-" + str(CONST_size_image_algorithm) + "-" + str(CONST_num_epochs) + ".prm")
        else:
            print "[ERRO]: Model do not exist!"
            print "[ERRO]: Please create a new model."
            sys.exit(-1)

    # Test Section


    l_out = cnn_functs.do_tests(test_set, model)

    # Analyze the results
    cnn_functs.analyze_results(l_out, figure_test_set)


    end_program = time.time()
    print "\tTime spend to organize: ", (end_program - start_program) / 60, 'minutes, or ', ((end_program - start_program) / 60) / 60, " hours\n"

    print "\tEND OF EXECUTION\n"

main()