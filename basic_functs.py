
from settings import *

import time
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
import cPickle

# crop and resize to 32x32
from PIL import Image
import numpy as np
from random import randrange
from neon.data import ArrayIterator


def save_results(l_out):
    print "[INFO]: Saving the Output"
    f = open("./out/l_out.txt", "w+")

    i = 0
    for batch in l_out:
        f.write(str(i) + "\n")
        for region in batch:
            f.write("\t" + str(region) + "\n")

        f.write("\n")

        i += 1

    f.close()


def making_arrayIterator(l_np_faces, l_np_non_faces, const_size_image):

    start = time.time()

    # Create a unique list with each np_image generate
    np_data_set = np.concatenate((l_np_faces, l_np_non_faces))
    np_data_set = np.asarray(np_data_set, dtype=np.uint8)

    # Create the labels
    np_label_set = np.concatenate(([1] * len(l_np_faces), [0] * len(l_np_non_faces)))

    print "\n\tData_set size  (quant, size): ", np_data_set.shape
    print "\tLabel_set size (quant,):      ", np_label_set.shape, "\n"

    global NUM_IMAGES_TRAINED
    NUM_IMAGES_TRAINED = np_label_set.shape[0]

    print "[INFO]: Creating the ArrayIterator of training set"

    # Create the array iterator with information
    train_set = ArrayIterator(X=np_data_set, y=np_label_set, nclass=2, lshape=(1, const_size_image, const_size_image))

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return train_set


def load_fddb(directories, size_image):

    l_class_Figure = []

    start = time.time()

    # For each directory specified
    for directory in directories:

        # Open the file of directories
        file_features = open(directory, "r")

        # Create a list of diretories
        s_line = file_features.readline()

        # Read them
        while s_line != "":

            # Instancie a new Class
            l_class_Figure.append(Figura_Class())

            # Read the s_path of np_image
            l_class_Figure[-1].set_path(s_line[:len(s_line) - 1] + ".jpg")

            np_image = load_image("./data_sets/originalPics/" +
                                        l_class_Figure[-1].get_path())

            # Load the np_image
            l_class_Figure[-1].set_image(np_image)

            # basic_functs.show_img(l_class_Figure[-1].get_image())

            # Read quantity of faces
            s_line = file_features.readline()

            int_number_faces = int(s_line)

            l_class_Figure[-1].set_number_faces(int_number_faces)

            # Read the informations of each face
            for index_face in range(0, int_number_faces):
                # Read the line and cut everthing it is not necessary
                s_line = file_features.readline()

                # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
                features_one_figure = s_line.split(" ")

                features_one_figure_cut = \
                    features_one_figure[:len(features_one_figure) - 2]

                # Add the new face in object
                l_class_Figure[-1].add_face_position(
                    features_one_figure_cut)

            s_line = file_features.readline()

        # Close the actual file
        file_features.close()

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return l_class_Figure


def generate_faces_fddb(l_class_Figure, size_image):

    # Create a list of faces and non_faces
    l_faces = []

    # Start tue cutting

    print "[INFO]: Cutting the images, getting the faces"

    start = time.time()

    # For each np_image
    for i in range(0, len(l_class_Figure)):
        # Printe some informations
        sys.stdout.write("\r\tProcessed: " + str(i + 1) + " of " + str(len(l_class_Figure)) + ". \tCompleted: " +
                         str((i + 1) / float(len(l_class_Figure)) * 100.0) + "%")

        sys.stdout.flush()

        #       0                 1             2       3        4
        # <major_axis_radius minor_axis_radius angle center_x center_y>

        # Load the information from this np_image i
        features = l_class_Figure[i].get_face_positions()
        np_img = l_class_Figure[i].get_image()

        j = 0
        # for each face in the np_image
        for feature in features:

            # Convert the string to int
            int_feature = [int(float(x)) for x in feature]

            # Verify if np_image pass the edge of np_image
            if int_feature[4] + int_feature[0] > np_img.shape[0]:
                int_feature[4] -= int_feature[4] + int_feature[0] - (np_img.shape[0] - 1)

            if int_feature[3] + int_feature[0] > np_img.shape[1]:
                int_feature[3] -= int_feature[3] + int_feature[0] - (np_img.shape[1] - 1)

            # Verify if np_image pass the edge of np_image
            if int_feature[4] - int_feature[0] < 0:
                int_feature[4] += abs(int_feature[4] + int_feature[0])

            if int_feature[3] - int_feature[0] < 0:
                int_feature[3] += abs(int_feature[3] + int_feature[0])

            # Calculate the interval where the face it is.
            y = [int_feature[4] - int_feature[0], int_feature[4] + int_feature[0]]
            x = [int_feature[3] - int_feature[0], int_feature[3] + int_feature[0]]

            # Cut
            np_img_cut = cut_and_resize_img(x, y, np_img, size_image)

            # basic_functs.show_img(np_img_cut, const_size_image)

            save_image(np_img_cut, "./train/face/f" + str(i) + "-" +
                                    str(j) + ".jpg",
                                    size_image, size_image)

            # Save the new face img
            l_faces.append(np_img_cut)

            j += 1


    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return l_faces


def generate_non_faces_fddb(l_class_Figure, size_image):

    l_np_non_faces = []

    # Start tue cutting

    print "[INFO]: Cutting the images, getting the non-faces"

    start = time.time()

    # For each np_image
    for i in range(0, len(l_class_Figure)):

        # Printe some informations
        sys.stdout.write("\r\tProcessed: " + str(i + 1) + " of " +
                         str(len(l_class_Figure)) +
                         ". \tCompleted: " +
                         str((i + 1) / float(len(l_class_Figure)) * 100.0) + "%")
        sys.stdout.flush()

        # for each face in the np_image
        for generations in range(0, 10):

            np_img = l_class_Figure[i].get_image()

            shift_x = randrange(np_img.shape[1] - size_image / 2) + size_image
            shift_y = randrange(np_img.shape[0] - size_image / 2) + size_image

            y_non_face = [shift_y - size_image, shift_y + size_image]
            x_non_face = [shift_x - size_image, shift_x + size_image]

            np_img_cut = cut_and_resize_img(x_non_face, y_non_face, np_img, size_image)

            # basic_functs.show_img(img_cut)
            save_image(np_img_cut, "./train/non_face/n" + str(i) + "-" + str(generations) + ".jpg", size_image, size_image)

            # Save the new non_face np_image
            l_np_non_faces.append(np_img_cut)

    # Calcule the time
    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    return l_np_non_faces


def load_94_and_generate_faces(num_folders, size_image):

    l_class_Figure = []

    l_np_faces = []

    s_path = "./data_sets/faces94/"

    l_up_folders = os.listdir(s_path)

    l_up_folders = l_up_folders[1:]

    for l_down_folders in l_up_folders:

        l_image_folders = os.listdir(s_path + l_down_folders)

        l_image_folders = l_image_folders[1:]

        if num_folders != 0:
            if num_folders <= len(l_image_folders):
                l_image_folders = l_image_folders[1:num_folders + 1]
            else:
                l_image_folders = l_image_folders[1:len(l_image_folders)]
        else:
            l_image_folders = l_image_folders[1:]

        for folder in l_image_folders:
            l_images_path = os.listdir(s_path + l_down_folders + "/" + folder + "/")

            i = 0
            for name_image in l_images_path:

                if ".gif" in name_image:
                    break

                if name_image[0] != '.':
                    # Instancie a new Class
                    l_class_Figure.append(Figura_Class())

                    # Read the s_path of np_image
                    l_class_Figure[-1].set_path(s_path + l_down_folders + "/" + folder + "/" + name_image)

                    # Load the np_image
                    np_image = load_image(s_path + l_down_folders + "/" + folder + "/" + name_image)

                    np_image = np_image[15:185, 20:160]

                    np_image = resize_img(np_image, size_image)

                    # show_img(np_image, size_image)

                    save_image(np_image, "./train/face/" + folder + "-" + str(i) + "-" + str(randrange(1, stop=10000)) + ".jpg", size_image, size_image)

                    l_class_Figure[-1].set_image(np_image)

                    l_np_faces.append(np.array(np_image, dtype=np.uint8).reshape(-1))

                    i += 1

                    if i == 20:
                        break

    return l_class_Figure, l_np_faces


def show_img(np_img, x=None, y=None):
    """
    Function that show the np_image without save it.
    :param img: np_image that will backend showed
    :param size_image: size of np_image
    :return: do not have a return
    """

    if y == None:
        y = x

    """
    # Convert the img to a numpy
    if "numpy" not in str(type(img)):
        np_img = np.array(img)
    else:
        np_img = img
    """


    # Verify if it is a vector or a matrix
    if len(np_img.shape) == 1:
        # If it is a vector, reshape to a matrix
        img_reshaped = np_img.reshape((x, y))
    else:
        img_reshaped = np_img

    print img_reshaped.shape

    # Convert the numpy to a Image
    img_out = Image.fromarray(np.asarray(img_reshaped, dtype=np.uint8), "L")

    # Do the show
    img_out.show()


def cut_and_resize_img(x, y, img_original, size_image):
    """
    Procedure that receive one np_image, cut and resize them the a new size.
    :param x: Intervals of X
    :param y: Intervals of Y
    :param img_original: Image that will backend re-sized
    :param size_image: The new format of np_image
    :return: Return the new np_image cut and re-sized
    """
    img = img_original

    # Generate the intervals
    #x_range = range(x[0], x[1])
    #y_range = range(y[0], y[1])

    # Cut the np_image
    # matrix_cropped = img[y_range][:, x_range][:]
    matrix_cropped = img[y[0]: y[1], x[0]: x[1]]

    # Transform to a Image Object
    img_thumbnail = Image.fromarray(np.asarray(matrix_cropped, dtype=np.uint8), "L")

    # Resize the np_image
    img_thumbnail = img_thumbnail.resize((size_image, size_image), Image.ANTIALIAS)

    # Return the new np_image
    return np.array(img_thumbnail, dtype=np.uint8).reshape(-1)


def resize_img(img, size_image):
    """
    Receive a np_image from parameter and resize it to the CONST_size_image_algorithm
    :param img: Image
    :param size_image: new format
    :return:
    """

    new_img = Image.fromarray(img, "L")

    # Resize it
    crop_img = new_img.resize((size_image, size_image), Image.ANTIALIAS)

    # Return the new np_image
    return np.array(crop_img, dtype=np.uint8).reshape(-1)


def unpickle(directory):
    """
    Function to unpack the dataset
    :param directory: Directory to dataset
    :return: Return a dictionary
    """
    fo = open(directory, "rb")
    dict_loading = cPickle.load(fo)
    fo.close()
    return dict_loading


def load_image(directory):
    """
    Function to read a unique np_image from directory
    :param directory: Directory to np_image
    :return: np_image in format numpy
    """
    img_loading = Image.open(directory).convert('L')
    img_loading.load()

    return np.asarray(img_loading, dtype=np.uint8)


def save_image(np_img, directory, lin=None, col=None):
    """
    Save a np_image in a specific directory
    :param img: the np_image
    :param directory: the directory
    :param lin: rows of np_image
    :param col: columns of np_image
    """

    """
    # Convert the img to a numpy
    if "numpy" not in str(type(img)):
        img_np = np.array(img)
    else:
        img_np = img
    """

    # Verify if it np_image is on the correct shape
    if len(np_img.shape) == 1:
        if lin == None:
            print "ERROR, Defina o valor lin ou col"
            sys.exit(88)

        img_reshaped = np_img.reshape((lin, col))

    else:
        img_reshaped = np_img

    # Save the np_image
    img_out = Image.fromarray(np.asarray(img_reshaped, dtype=np.uint8), "L")

    img_out.save(directory)


def verify_args():
    """
    Procedure to verify the arguments from terminal
    :return:
    """
    # TODO asfk
    # if len(sys.argv) == 2:
    if True:
        # do_again = int(sys.argv[1])
        train_again = 1
        if train_again != 0 and train_again != 1:
            print "[ERRO]: Invalid Parameters."
            print "[INFO]: Use name_of_program.py option_training."
            print "[INFO]: option_training: 1 - create a new " \
                  "Neural Network; 0 - Use already saved."
            sys.exit()
    else:
        print "[ERRO]: Invalid Parameters."
        print "[INFO]: Use name_of_program.py option_training."
        print "[INFO]: option_training: 1 - create a new Neural " \
              "Network; 0 - Use already saved."
        sys.exit()

    return train_again
