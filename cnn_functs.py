# Search files on path
import os.path
#
import sys
import os
import shutil

from neon.data import CIFAR10, ArrayIterator
from neon.layers import Conv, Affine, Pooling, GeneralizedCost
from neon.initializers import Uniform
from neon.transforms.activation import Rectlin, Softmax
from neon.transforms import CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.callbacks.callbacks import Callbacks
from random import randrange
import time
import numpy as np
import basic_functs


class Figura_Class:
    """
    Class that save the informations of a image
    """
    description = "<major_axis_radius minor_axis_radius angle center_x center_y>"

    def __init__(self, path="", number_faces=0):
        self.path = path
        self.number_faces = number_faces
        self.faces_positions = []
        self.image = []

    def get_path(self):
        return self.path

    def set_path(self, path):
        self.path = path

    # Nun_Faces

    def get_number_faces(self):
        return self.number_faces

    def set_number_faces(self, number_faces):
        self.number_faces = number_faces

    # Image

    def get_image(self):
        return self.image

    def set_image(self, image):
        self.image = image

    # Face

    def add_face_position(self, vector):
        self.faces_positions.append(vector)

    def get_face_positions(self):
        return self.faces_positions


def loading_set_for_training(diretorios, size_image):
    """
    Procedure to read dataset for training.
    :param diretorios: directorys for read the images. It can be more than one.
    :param size_image: Size of images to train
    :return: A lot of things
    """
    print "[INFO]: Loading training FDDB"

    # list of new figures and its informations
    l_figure_class = []
    # Number of figures read
    number_figures = 0
    # Coeficient of difference between the face and non-face
    diff = 60

    # Calcule the time
    start = time.time()

    # Verify if exist a path to save images, if needed
    if os.path.exists("./train/"):
        shutil.rmtree("./train/")

    # Make the directorys
    os.mkdir("./train/")
    os.mkdir("./train/face/")
    os.mkdir("./train/non_face/")

    # For each directory specified
    for diretorio in diretorios:

        # Open the file of directories
        file_features = open(diretorio, "r")

        # Create a list of diretories
        s_line = file_features.readline()

        # Read them
        while s_line != "":

            # Instancie a new Class
            l_figure_class.append(Figura_Class())

            # Read the path of image
            l_figure_class[number_figures].set_path(s_line[:len(s_line) - 1] + ".jpg")

            # Load the image
            l_figure_class[number_figures].set_image(
                basic_functs.load_image("./data_sets/originalPics/" +
                                        l_figure_class[number_figures].get_path()))

            # Verify if it image is colored
            if len(l_figure_class[number_figures].get_image().shape) == 3:

                # Read quantity of faces
                s_line = file_features.readline()

                number_faces = int(s_line)

                l_figure_class[number_figures].set_number_faces(number_faces)

                # Read the informations of each face
                for index_face in range(0, number_faces):

                    # Read the line and cut everthing it is not necessary
                    s_line = file_features.readline()

                    # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
                    l_features_of_one_figure_buffer = s_line.split(" ")

                    l_features_of_one_figure_buffer_cut = \
                        l_features_of_one_figure_buffer[:len(l_features_of_one_figure_buffer) - 2]

                    # Add the new face in object
                    l_figure_class[number_figures].add_face_position(l_features_of_one_figure_buffer_cut)

                # Count the figures
                number_figures += 1

                s_line = file_features.readline()

            # If the figure is black and white, delete it and pass to next.
            else:

                del l_figure_class[number_figures]

                s_line = file_features.readline()

                for i in range(int(s_line) + 1):
                    s_line = file_features.readline()

        # Close the actual file
        file_features.close()

    # Create a list of faces and non_faces
    l_faces = []
    l_non_faces = []

    # Calcule the time
    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    # Start tue cutting

    print "[INFO]: Cutting the images, getting the faces and non-faces"

    start = time.time()

    # For each image
    for i in range(0, number_figures):
        # Printe some informations
        sys.stdout.write("\r\tProcessed: " + str(i + 1) + " of " + str(number_figures) +
                         ". \tCompleted: " + str((i+1) / float(number_figures) * 100.0 ) + "%")
        sys.stdout.flush()

        #       0                 1             2       3        4
        # <major_axis_radius minor_axis_radius angle center_x center_y>

        # Load the informations from this image i
        features = l_figure_class[i].get_face_positions()
        img = l_figure_class[i].get_image()

        # for each face in the image
        for feature in features:

            # Convert the string to int
            feature_int = [int(float(x)) for x in feature]

            # Verify if image pass the edge of image
            if feature_int[4] + feature_int[0] > img.shape[0]:
                feature_int[4] -= feature_int[4] + feature_int[0] - (img.shape[0] - 1)

            if feature_int[3] + feature_int[0] > img.shape[1]:
                feature_int[3] -= feature_int[3] + feature_int[0] - (img.shape[1] - 1)

            # Calculate the interval where the face it is.
            y = [feature_int[4] - feature_int[0], feature_int[4] + feature_int[0]]
            x = [feature_int[3] - feature_int[0], feature_int[3] + feature_int[0]]

            # Cut
            img_cut = basic_functs.cut_and_resize_img(x, y, img, size_image)

            # basic_functs.show_img(img_cut)
            #basic_functs.save_image(img_cut, "./train/face/f" + str(randrange(1, stop=10000)) + ".jpg",
            #                        size_image, size_image)

            # Save the new face img
            l_faces.append(img_cut)

            # Creating new images non-face for training:
            for i in range(2):
                #       0                 1             2       3        4
                # <major_a xis_radius minor_axis_radius angle center_x center_y>

                shift_x = randrange(img.shape[1] - feature_int[0])
                shift_y = randrange(img.shape[0] - feature_int[0])

                # Verify is the new image is too near to face original one
                while abs(shift_x - feature_int[3]) < diff or abs(shift_y - feature_int[4]) < diff:
                    if abs(shift_x - feature_int[3]) < diff:
                        shift_x = randrange(img.shape[1] - feature_int[0])
                    else:
                        shift_y = randrange(img.shape[0] - feature_int[0])

                y_non_face = [shift_y - feature_int[0], shift_y + feature_int[0]]
                x_non_face = [shift_x - feature_int[0], shift_x + feature_int[0]]

                img_cut = basic_functs.cut_and_resize_img(x_non_face, y_non_face, img,
                                                          size_image)

                # basic_functs.show_img(img_cut)
                # basic_functs.save_image(img_cut, "./train/non_face/n" + str(randrange(1, stop=10000)) + ".jpg",
                #                        size_image, size_image)

                # Save the new non_face image
                l_non_faces.append(img_cut)

    print "\n\n[INFO]: Read and Cutting Faces and Non-Faces Done"

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    # Create a unique list with each image generate
    np_data_set = np.concatenate((l_faces, l_non_faces))
    np_data_set = np.asarray(np_data_set)

    # Create the labels
    np_label_set = np.concatenate(([1] * len(l_faces), [0] * len(l_faces) * 2))

    print "\tData_set size  (quant, size): ", np_data_set.shape
    print "\tLabel_set size (quant,):      ", np_label_set.shape, "\n"

    print "[INFO]: Creating the ArrayIterator of training set"

    # Create the array iterator with informations
    train_set = ArrayIterator(X=np_data_set,
                              y=np_label_set, nclass=2, lshape=(3, size_image,
                                                                size_image))

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return train_set, l_figure_class


def loading_set_for_testing(diretorios):
    """
    Procedure that loads the dataset for testing
    :param diretorios: List of directors
    :return: return the images.
    """
    print "[INFO]: Loading test FDDB"

    start = time.time()

    l_figure_class = []

    # For each directory
    for diretorio in diretorios:

        # Opening the file
        f_paths = open(diretorio, "r")

        s_line = f_paths.readline()

        # For each path
        while s_line != "":
            # Create a new Figura Class
            l_figure_class.append(Figura_Class())

            # Save the path
            l_figure_class[-1].set_path(s_line[:len(s_line) - 1] + ".jpg")

            # Save the image
            l_figure_class[-1].set_image(basic_functs.load_image("./data_sets/originalPics/" +
                                                                l_figure_class[-1].get_path()))

            # If the image are black and white, delete it and jump to next
            if len(l_figure_class[-1].get_image().shape) == 2:
                del l_figure_class[-1]

            s_line = f_paths.readline()

        f_paths.close()

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return l_figure_class


def making_regions(l_Figure_class, batch_size, size_image):
    """
    Procedure that gets the figures test and create a lot of regions to do the tests
    :param l_Figure_class: List of figures
    :param batch_size: Size of batch
    :param size_image: Size of each image to test
    :return: The regions list
    """
    print "[INFO]: Generating the Regions"

    start = time.time()
    l_batch = []

    # For each Image
    for Figura_original in l_Figure_class:

        # Bound of each figure to test
        bound = size_image
        # Distance's interval of the face
        interval = 90
        l_new_images = []

        # Quantity of new images created
        quant_new_images = 0

        # Verify if the region respect the bound of the face
        while bound < Figura_original.get_image().shape[0] or \
                        bound < Figura_original.get_image().shape[1]:

            # Move on the Y axis
            for edge_y_axis in range(bound, Figura_original.get_image().shape[0], interval):

                # Move on the X axis
                for edge_x_axis in range(bound, Figura_original.get_image().shape[1], interval):

                    # Crop the new image
                    crop_img = Figura_original.get_image()[edge_y_axis - bound: edge_y_axis
                               ][:, edge_x_axis - bound: edge_x_axis][:]

                    # Verify if the image is little than the bound of algorithm
                    if bound > size_image:
                        l_new_images.append(basic_functs.resize_img(crop_img, size_image))
                    else:
                        l_new_images.append(np.array(crop_img).reshape(-1))

                    quant_new_images += 1

                    # Add on the information of the region Figure list
                    Figura_original.faces_positions.append([edge_x_axis, edge_y_axis, bound])

                    # if the batch of this image list complete, stop to create new regions
                    if quant_new_images >= batch_size:
                        break

                # if the batch of this image list complete, stop to create new regions
                if quant_new_images >= batch_size:
                    break

            # if the batch of this image list complete, stop to create new regions
            if quant_new_images >= batch_size:
                break

            # if the batch of this image list complete, stop to create new regions
            bound += 120
            interval = bound - bound / 4

        l_batch.append(l_new_images)

    print "\tNumbers of Batches for cropped images:       ", len(l_batch)
    print "\t\tNumbers of cropped images of each image: ", len(l_batch[0])

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return np.array(l_batch)


def generate_inference(batches, batch_size, size_image):
    """
    Procedure that generates the inferences for test
    :param batches: list of batches of images
    :param batch_size: size of each batch
    :param size_image: size of images of algorithm
    :return: list of inference
    """

    print "[INFO]: Creating the ArrayIterators for test"

    start = time.time()

    l_inferences = []

    # For each batch from list
    i = 1
    for batch in batches:
        sys.stdout.write("\r\tProcessed: " + str(i) + " of " + str(len(batches)) +
                         ". \tCompleted: " +
                         str((i) / float(len(batches)) * 100.0 ) + "%.\tTime spend until now: " +
                         str(time.time() - start) + "s")
        sys.stdout.flush()

        # Create a new list empty
        x_new = np.zeros((batch_size, size_image * size_image * 3))

        # Add the batch to the list
        x_new[0:len(batch)] = np.asarray(batch)

        # Create the array iterator from the new list
        inference_out = ArrayIterator(X=x_new, y=None, nclass=2, lshape=(3, size_image, size_image))

        # Save in a list
        l_inferences.append(inference_out)

        i += 1

    end = time.time()
    print "\n\tTime spend to generate the inferences: ", end - start, 'seconds', "\n"

    return l_inferences


def test_inference(l_inferences, model):
    """
    Function that makes the tests
    :param l_inferences: list of inferences
    :param model: Model that contains the CNN
    :return: Result of test
    """

    print "[INFO]: Making the tests, please wait"

    start = time.time()
    l_out = []

    print "\tNumbers of Batchs with ArrayIterators: ", len(l_inferences)

    # For each inference
    i = 1
    for inference in l_inferences:
        sys.stdout.write("\r\tProcessed: " + str(i) + " of " + str(len(l_inferences)) +
                         ". \tCompleted: " + str((i) / float(len(l_inferences)) * 100.0) +
                         "%.\tTime spend until now: " + str(time.time() - start) + "s")
        sys.stdout.flush()

        # Do test
        out = model.get_outputs(inference)

        l_out.append(out)
        i += 1


    end = time.time()
    print "\n\tTime spend to generate the outputs: ", end - start, 'seconds', "\n"

    print "[INFO]: Tests Done"

    return l_out


# TODO marcar a foto
def mark_result(image, position):
    """
    Procedure that mark the result of process
    :param image: image to mark
    :param position: position to mark
    :param path: path to save
    :return: do not have return
    """

    new_img = np.array(image)
    coords = position

    edge_x_axis = coords[0]
    edge_y_axis = coords[1]
    bound = coords[2]

    # mark the y axis
    value = 255
    for y in range(edge_y_axis - bound, edge_y_axis):
        new_img[edge_x_axis, y, 0] = value
        new_img[edge_x_axis - bound, y, 0] = value

        new_img[edge_x_axis, y, 1:2] = 0
        new_img[edge_x_axis - bound, y, 1:2] = 0

    # Mark the x axis
    for x in range(edge_x_axis - bound, edge_x_axis):
        new_img[x, edge_y_axis, 0] = value
        new_img[x, edge_y_axis - bound, 0] = value

        new_img[x, edge_y_axis, 1:2] = 0
        new_img[x, edge_y_axis - bound, 1:2] = 0

    basic_functs.show_img(new_img, bound)

    return new_img


def analyze_results(l_out, l_Figura_class):
    print "[INFO]: Analyzing the Results"

    index = 0

    if os.path.exists("./out/"):
        shutil.rmtree("./out/")

    os.mkdir("./out/")
    os.mkdir("./out/f/")
    os.mkdir("./out/n/")

    # Para cada imagem 0.5 0.5
    for batch in l_out:

        # para cada regiao
        num_region = 0
        image_marked = False
        for region in batch:
            # se for encontrado algum rosto

            if region[0] > 0.5:
                print region, len(l_Figura_class[index].faces_positions), num_region
                print "."
                image_marked = True

                print "shape:", l_Figura_class[index].get_image(), "position:",l_Figura_class[index].faces_positions[num_region]

                img = mark_result(l_Figura_class[index].get_image(),
                               l_Figura_class[index].faces_positions[num_region])

                l_Figura_class[index].set_image(img)

            num_region += 1

            if image_marked:
                basic_functs.save_image(l_Figura_class[index].get_image(), "./out/f/f" + str(randrange(1, stop=10000)) + ".jpg")

        index += 1
    #sys.exit(11)
"""

#classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#print classes
#classes = ["No", "Yes"]

out = model.get_outputs(inference_set)

print out[0]
#print classes[out[0].argmax()]

"""