#!/usr/bin/python

from settings import *

from neon.backends import gen_backend
import os
from neon.models import Model
import sys
import time
import cnn_functs, basic_functs


class Figura_Class:
    """
    Class that save the images's data and organization
    """
    description = "<major_axis_radius minor_axis_radius angle center_x center_y>"

    def __init__(self, path="", number_faces=0):
        self.path = path                       # Path to the np_image
        self.number_faces = number_faces       # Number of Faces
        self.l_faces_positions_original = []   # The features of each position of face from file
        self.l_faces_positions_regions = []   # The features of each position of face from file
        self.image = []                        # The Image file

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

    def add_face_position_original(self, vector):
        self.l_faces_positions_original.append(vector)

    def get_face_positions_original(self):
        return self.l_faces_positions_original

    # Face positions

    def add_face_position_regions(self, vector):
        self.l_faces_positions_regions.append(vector)

    def get_face_positions_regions(self):
        return self.l_faces_positions_regions

NUM_IMAGES_TRAINED = -1


def main():

    basic_functs.verify_args()

    CONST_batch_size = sys.argv[1]            # 512
    CONST_num_epochs = sys.argv[2]            # 1250  # Passages on through the dataset
    CONST_size_image_algorithm = sys.argv[3]  # 36  # Size of training np_image
    CONST_learning_rate = sys.argv[4]         # 0.01
    CONST_momentum = sys.argv[5]              # 0.9
    CONST_train_again = sys.argv[6]           # "n"
    num_files_94 = sys.argv[7]                # 0


    l_directories_fddb_train = basic_functs.read_paths(sys.argv[8])

    l_directories_fddb_test = basic_functs.read_paths(sys.argv[9])

    l_directories_fddb_valid = basic_functs.read_paths(sys.argv[10])

    start_program = time.time()

    # Start
    print "[INFO]: Library read."
    print "[SETU]: Setting the CPU."

    # Starting the procedures of configurations

    backend = gen_backend(backend="cpu", batch_size=CONST_batch_size)
    print "[INFO]: Batch_size:", CONST_batch_size, "\tSize of the Images:", CONST_size_image_algorithm

    print "[INFO]: Information about backend:"
    print "\t", backend, "\n"

    # Load the images creating the training set
    if CONST_train_again == "y":
        train_set = cnn_functs.loading_set_for_training(num_files_94, l_directories_fddb_train, CONST_size_image_algorithm)

        test_set, figure_test_set = cnn_functs.loading_set_for_testing(l_directories_fddb_test, CONST_batch_size, CONST_size_image_algorithm)

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


    for list_images in l_directories_fddb_valid:
        valid_set, figure_valid_set = cnn_functs.loading_set_for_validation(list_images, CONST_batch_size, CONST_size_image_algorithm)

        # Test Section
        l_out = cnn_functs.do_validation(valid_set, model)

        # Analyze the results
        basic_functs.analyze_results(l_out, figure_valid_set, int(list_images[33:35]))


    end_program = time.time()
    print "\tTime spend to organize: ", (end_program - start_program) / 60, 'minutes, or ', ((end_program - start_program) / 60) / 60, " hours\n"

    print "\tEND OF EXECUTION\n"

main()