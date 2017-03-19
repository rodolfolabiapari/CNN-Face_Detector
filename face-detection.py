#!/usr/bin/python

class Figura_Class:
    """
    Class that save the images's data and organization
    """
    description = "<major_axis_radius minor_axis_radius angle center_x center_y>"

    def __init__(self, path="", number_faces=0):
        self.path = path                  # Path to the image
        self.number_faces = number_faces  # Number of Faces
        self.faces_positions = []         # The features of each position of face
        self.image = []                   # The Image file

    # Path

    def get_path(self):
        return self.path

    def set_path(self, path):
        self.path = path

    # Number of Faces

    def get_number_faces(self):
        return self.number_faces

    def set_number_faces(self, number_faces):
        self.number_faces = number_faces

    # Image

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    # Face positions

    def add_face_position(self, vector):
        self.faces_positions.append(vector)

    def get_face_positions(self):
        return self.faces_positions


# Importations
from neon.backends import gen_backend
import os.path
import sys
from neon.data import ArrayIterator
from neon.layers import Conv, Affine, Pooling, GeneralizedCost
from neon.initializers import Uniform, Gaussian
from neon.transforms.activation import Rectlin, Softmax
from neon.transforms import CrossEntropyMulti, CrossEntropyBinary, Misclassification
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.callbacks.callbacks import Callbacks

import urllib                    # download the image
from PIL import Image            # crop and resize to 32x32
import numpy as np               # Work with vectors and matrices
import time
import basic_functs
import cnn_functs


# Start
print "[INFO]: Library read."
print "[SETU]: Setting the CPU."


batch_size = 128        # The batch_size cannot be bigger than amount images
num_epochs = 5          # Passages on through the dataset
size_image = 50          # Size of training image
# TODO arrumar o args
train_again = basic_functs.verify_args()
train_again = "n"

be = gen_backend(backend="cpu", batch_size=batch_size)

print "[INFO]: Information about backend:"
print "\t", be, "\n"



# Verify if there is necessity of re-train
if train_again == "y":
    print "[DOWN]: Downloading the dataset of training."

    # TODO folders from terminal reading from file
    diretorios_train = [
        "./data_sets/FDDB-folds/FDDB-fold-01-ellipseList.txt",
        "./data_sets/FDDB-folds/FDDB-fold-02-ellipseList.txt",
        "./data_sets/FDDB-folds/FDDB-fold-03-ellipseList.txt",
        "./data_sets/FDDB-folds/FDDB-fold-04-ellipseList.txt"
        #"./data_sets/FDDB-folds/FDDB-fold-05-ellipseList.txt"
        # "./data_sets/FDDB-folds/FDDB-fold-01-ellipseList-1.txt"
    ]

    # Load the images creating the training set
    train_set, train_Figures = cnn_functs.loading_set_for_training(diretorios_train,
                                                                   size_image)

    print "[INFO]: The Neural Network was not created yet."
    print "[INFO]: Starting procedure of creating and training."

    # Procedure of Costs
    #     Setting up the cost function of network output

    print "[INFO]: Creating the cost function."
    cost = GeneralizedCost(costfunc=CrossEntropyBinary())

    # Procedure for Build Model

    print "[INFO]: Making the layers."

    # Network with two Conv, two Pooling, and two Affine layers.

    print "\tCreating a Convolutional Network.\n"
    init_uni = Uniform(low=-0.1, high=0.1)

    layers = [Conv(fshape=(5, 5, 16),
                   init=init_uni,
                   activation=Rectlin(), padding=True),

              Pooling(fshape=2, strides=2),

              Conv(fshape=(5, 5, 32),
                   init=init_uni,
                   activation=Rectlin(), padding=True),

              Pooling(fshape=2, strides=2),

              Affine(nout=128,
                     init=init_uni,
                     bias=init_uni,
                     activation=Rectlin()),

              Affine(nout=2,
                     init=init_uni,
                     activation=Softmax())]

    # Optimizer

    print "[INFO]: Setting the optimizer with values: " \
          "learning_rate=0.005, momentum_coef=0.9."
    # Optimizer
    #    Having the cost function, we want minimize it.
    #optimizer = GradientDescentMomentum(learning_rate=0.005, momentum_coef=0.9)
    optimizer = GradientDescentMomentum(learning_rate=0.5, momentum_coef=0.9)


    print "[SETU]: Setting the layers up."
    # Seting up of the model
    model = Model(layers)

    # Callback

    print "[INFO]: Creating the Callbacks."
    callbacks = Callbacks(model, train_set)

    # Training

    print "[INFO]: Making the Neural Network with", num_epochs, "epochs."

    start = time.time()
    model.fit(dataset=train_set, cost=cost, optimizer=optimizer,
              num_epochs=num_epochs, callbacks=callbacks)
    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'
    print "\tTime spend to organize: ", (end - start) / 60.0, 'minutes\n'

    # Saving

    print "[BACK]: Saving the model with the name \"cnn-trained_model.prm\".\n\n"
    model.save_params("cnn-trained_model.prm")

else:
    print "[INFO]: Network already created."
    print "[LOAD]: Loading the model with the name \"cnn-trained_model.prm\".\n\n"

    if os.path.exists("./cnn-trained_model.prm"):
        model = Model("cnn-trained_model.prm")
    else:
        print "[ERRO]: Model do not exist!"
        print "[ERRO]: Please create a new model."


# Test Section

# TODO read files
diretorios_test = [#"./data_sets/FDDB-folds/FDDB-fold-01-2.txt"
                   "./data_sets/FDDB-folds/FDDB-fold-01-1.txt"
                   #"./data_sets/FDDB-folds/FDDB-fold-01.txt"
                   ]

# Create a list of figures from test dataset
test_Figures = cnn_functs.loading_set_for_testing(diretorios_test)

# Create a list of batches for test
l_batches_test = cnn_functs.making_regions(test_Figures, batch_size, size_image)

# Generate the inference lists for tests
test_set = cnn_functs.generate_inference(l_batches_test, batch_size, size_image)

# Calcule the Miss classification error by framework
miss_test = False or True # todo retirar essa variavel e colocar no argumentos
if miss_test:
    print "[INFO]: Checking the Misclassification of error."
    start = time.time()
    error_pct = 100 * model.eval(test_set[0], metric=Misclassification())
    end = time.time()
    print "\tTime spend to organize:   ", end - start, 'seconds'
    print "\tMiss classification error: %.3f%%" % error_pct, "\n"

# Test each one batch from list of batches
l_out = cnn_functs.test_inference(test_set, model)

# Analyze the results
cnn_functs.analyze_results(l_out, test_Figures)