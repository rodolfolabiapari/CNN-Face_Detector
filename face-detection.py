#!/usr/bin/python

class Figura_Class:
    """
    Class that save the images's data and organization
    """
    description = "<major_axis_radius minor_axis_radius angle center_x center_y>"

    def __init__(self, path="", number_faces=0):
        self.path = path                  # Path to the np_image
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
from neon.transforms import Misclassification
from neon.models import Model

import urllib                    # download the np_image
from PIL import Image            # crop and resize to 32x32
import numpy as np               # Work with vectors and matrices
import time
import basic_functs
import cnn_functs

NUM_IMAGES_TRAINED = -1
CONST_batch_size = 4096          # The CONST_batch_size cannot backend bigger
# than amount images TODO DEVE SER AUTO
# CONST_batch_size = 3084
# CONST_batch_size = 2048
#CONST_batch_size = 1024
# CONST_batch_size = 512
#CONST_batch_size = 90
CONST_num_epochs = 2  # Passages on through the dataset
CONST_size_image_algorithm = 100  # Size of training np_image
CONST_learning_rate = 0.005
CONST_momentum = 0.9
# CONST_train_again = basic_functs.verify_args()       # TODO arrumar o args
CONST_train_again = "y"
# TODO
num_files_94 = 0


def main():
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
         "./data_sets/FDDB-folds/FDDB-fold-04-ellipseList.txt"
        #"./data_sets/FDDB-folds/FDDB-fold-01-ellipseList-1.txt"
    ]

    # Verify if there is necessity of re-train
    if CONST_train_again == "y":

        model = cnn_functs.train_model_fddb_94(num_files_94, l_directories_fddb_train, CONST_size_image_algorithm,
                                               CONST_num_epochs, CONST_learning_rate,
                                               CONST_momentum)
    else:
        print "[INFO]: Network already created."
        print "[LOAD]: Loading the model with the name \"cnn-trained_model.prm\".\n\n"

        if os.path.exists("cnn-trained_model-" + str(num_files_94) + "-" + str(len(l_directories_fddb_train)) + "-" +
                                  str(CONST_size_image_algorithm) + "-" +
                                  str(CONST_num_epochs) + ".prm"):

            model = Model("cnn-trained_model-" + str(num_files_94) + "-" + str(len(l_directories_fddb_train)) + "-" +
                          str(CONST_size_image_algorithm) + "-" +
                          str(CONST_num_epochs) + ".prm")
        else:
            print "[ERRO]: Model do not exist!"
            print "[ERRO]: Please create a new model."
            sys.exit(-1)

    # Test Section

    # TODO read files
    diretorios_test = ["./data_sets/FDDB-folds/FDDB-fold-05 copy.txt"
                       ]

    l_out, test_Figures = cnn_functs.do_tests(
        diretorios_test, CONST_batch_size, CONST_size_image_algorithm, model)

    # Analyze the results
    cnn_functs.analyze_results(l_out, test_Figures)


main()