# Search files on path
import os.path
#
import sys
import os
import shutil

from neon.data import CIFAR10, ArrayIterator
from random import randrange
import time
import numpy as np
import basic_functs
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


class Figura_Class:
    """
    Class that save the informations of a np_image
    """
    description = "<major_axis_radius minor_axis_radius angle center_x center_y>"

    def __init__(self, path="", number_faces=0):
        self.s_path = path
        self.int_number_faces = number_faces
        self.l_faces_positions = []
        self.np_image = []

    def get_path(self):
        return self.s_path

    def set_path(self, path):
        self.s_path = path

    # Nun_Faces

    def get_number_faces(self):
        return self.int_number_faces

    def set_number_faces(self, number_faces):
        self.int_number_faces = number_faces

    # Image

    def get_image(self):
        return self.np_image

    def set_image(self, image):
        self.np_image = image

    # Face

    def add_face_position(self, vector):
        self.l_faces_positions.append(vector)

    def get_face_positions(self):
        return self.l_faces_positions



def train_model_fddb_94(num_files_94, l_directories_fddb_train,  const_size_image, num_epochs, learning_rate, momentum):
    print "[DOWN]: Getting the dataset of training."

    # Load the images creating the training set
    train_set = loading_set_for_training(num_files_94, l_directories_fddb_train, const_size_image)

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

    layers = [Conv(fshape=(32, 32, 12),
                   init=init_uni,
                   activation=Rectlin(), padding=True),

              Pooling(fshape=2, strides=2),

              Conv(fshape=(16, 16, 36),
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
          "CONST_learning_rate=" + str(learning_rate) + \
          ", momentum_coef=" + str(momentum) + "."
    # Optimizer
    #    Having the cost function, we want minimize it.
    optimizer = GradientDescentMomentum(learning_rate=learning_rate, momentum_coef=momentum)

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
    model.save_params("cnn-trained_model-"+ str(num_files_94) + "-" + str(len(l_directories_fddb_train)) + "-" +
                      str(const_size_image) + "-" +
                      str(num_epochs) + ".prm")

    return model


def train_model(directories_train, const_size_image, num_epochs, learning_rate, momentum):
    print "[DOWN]: Downloading the dataset of training."

    # Load the images creating the training set
    train_set = loading_set_for_training(directories_train, const_size_image)

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

    layers = [Conv(fshape=(12, 12, 24),
                   init=init_uni,
                   activation=Rectlin(), padding=True),

              Pooling(fshape=2, strides=2),

              Conv(fshape=(10, 10, 48),
                   init=init_uni,
                   activation=Rectlin(), padding=True),

              Pooling(fshape=2, strides=2),

              Affine(nout=500,
                     init=init_uni,
                     bias=init_uni,
                     activation=Rectlin()),

              Affine(nout=2,
                     init=init_uni,
                     activation=Softmax())]

    # Optimizer

    print "[INFO]: Setting the optimizer with values: " \
          "CONST_learning_rate=" + str(learning_rate) + \
          ", momentum_coef=" + str(momentum) + "."
    # Optimizer
    #    Having the cost function, we want minimize it.
    optimizer = GradientDescentMomentum(learning_rate=learning_rate, momentum_coef=momentum)

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
    model.save_params("cnn-trained_model-"+ str(directories_train) + "-" +
                      str(const_size_image) + "-" +
                      str(num_epochs) + ".prm")

    return model


def do_tests(directories_test, batch_min_size, const_size_image, model):

    print "----------------------------------------------" \
          "\n[INFO]: Starting the procedure of tests."

    # Create a list of figures from test dataset
    test_figures = loading_set_for_testing(directories_test)

    # Create a list of batches for test
    l_batches_test = making_regions(test_figures, batch_min_size, const_size_image)

    # Generate the inference lists for tests
    test_set = generate_inference(l_batches_test, batch_min_size, const_size_image)

    # Calcule the Miss classification error by framework
    miss_test = False #or True  # todo retirar essa variavel e colocar no argumentos
    if miss_test:
        print "[INFO]: Checking the Miss classification of error."
        start = time.time()
        error_pct = 100 * model.eval(test_set[0], metric=Misclassification())
        end = time.time()
        print "\tTime spend to organize:   ", end - start, 'seconds'
        print "\tMiss classification error: %.3f%%" % error_pct, "\n"

    # Test each one batch from list of batches
    l_out = test_inference(test_set, model)

    return l_out, test_figures


def loading_set_for_training(num_files_94, l_directories_fddb_train, const_size_image):
    """
    Procedure to read dataset for training.
    :param directorys: directorys for read the images. It can backend more than one.
    :param const_size_image: Size of images to train
    :return: A lot of things
    """


    # Calcule the time
    start = time.time()

    # Verify if exist a s_path to save images, if needed
    if os.path.exists("./train/"):
        shutil.rmtree("./train/")

    # Make the directorys
    os.mkdir("./train/")
    os.mkdir("./train/face/")
    os.mkdir("./train/non_face/")

    # 94

    print "[INFO]: Loading training 94 data-set"
    l_class_Figure_94, l_np_faces_94 = basic_functs.load_94_and_generate_faces(num_files_94, const_size_image)

    print "\tNumber of Faces:     ", len(l_np_faces_94), "\n"
    # FDDB

    print "[INFO]: Loading training FDDB data-set"
    l_class_Figure_fddb = basic_functs.load_fddb(l_directories_fddb_train,
                                                       const_size_image)

    l_np_non_faces_fddb = basic_functs.generate_non_faces_fddb(l_class_Figure_fddb, const_size_image)

    if True:
        l_np_faces_fddb = basic_functs.generate_faces_fddb(l_class_Figure_fddb, const_size_image)
        l_np_faces = np.concatenate((l_np_faces_fddb, l_np_faces_94))

        print "\tNumber of Faces:     ", len(l_np_faces_fddb)
    else:
        l_np_faces = l_np_faces_94

    print "\tNumber of Non-Faces: ", len(l_np_non_faces_fddb), "\n"

    l_np_non_faces = l_np_non_faces_fddb

    print "[INFO]: Images read successfully"
    print "\tNumber of Faces:     ", len(l_np_faces)
    print "\tNumber of Non-Faces: ", len(l_np_non_faces)

    arrayIterator_train_set = basic_functs.making_arrayIterator(l_np_faces, l_np_non_faces, const_size_image)

    return arrayIterator_train_set

    """

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
                l_class_Figure[int_number_figures].set_path(s_line[:len(s_line) - 1] + ".jpg")

                # Load the np_image
                l_class_Figure[int_number_figures].set_image(
                    basic_functs.load_image("./data_sets/originalPics/" +
                                            l_class_Figure[int_number_figures].get_path()))

                # basic_functs.show_img(l_class_Figure[int_number_figures].get_image())


                # Verify if it np_image is colored
                if len(l_class_Figure[int_number_figures].get_image().shape) == 3:

                    # Read quantity of faces
                    s_line = file_features.readline()

                    int_number_faces = int(s_line)

                    l_class_Figure[int_number_figures].set_number_faces(int_number_faces)

                    # Read the informations of each face
                    for index_face in range(0, int_number_faces):

                        # Read the line and cut everthing it is not necessary
                        s_line = file_features.readline()

                        # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
                        features_one_figure = s_line.split(" ")

                        features_one_figure_cut = \
                            features_one_figure[:len(features_one_figure) - 2]

                        # Add the new face in object
                        l_class_Figure[int_number_figures].add_face_position(
                            features_one_figure_cut)

                    # Count the figures
                    int_number_figures += 1

                    s_line = file_features.readline()

                # If the figure is black and white, delete it and pass to next.
                else:

                    del l_class_Figure[int_number_figures]

                    s_line = file_features.readline()

                    for i in range(int(s_line) + 1):
                        s_line = file_features.readline()

            # Close the actual file
            file_features.close()

        # Create a list of faces and non_faces
        l_np_faces = []
        l_np_non_faces = []

        # Calcule the time
        end = time.time()
        print "\tTime spend to organize: ", end - start, 'seconds', "\n"

        # Start tue cutting

        print "[INFO]: Cutting the images, getting the faces and non-faces"

        start = time.time()

        # For each np_image
        for i in range(0, int_number_figures):
            # Printe some informations
            sys.stdout.write("\r\tProcessed: " + str(i + 1) + " of " +
                             str(int_number_figures) +
                             ". \tCompleted: " +
                             str((i+1) / float(int_number_figures) * 100.0) + "%")
            sys.stdout.flush()

            #       0                 1             2       3        4
            # <major_axis_radius minor_axis_radius angle center_x center_y>

            # Load the informations from this np_image i
            features = l_class_Figure[i].get_face_positions()
            np_img = l_class_Figure[i].get_image()

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
                np_img_cut = basic_functs.cut_and_resize_img(x, y, np_img, const_size_image)

                # basic_functs.show_img(np_img_cut, const_size_image)

                basic_functs.save_image(np_img_cut, "./train/face/f" + str(i) + "-" +
                                        str(randrange(1, stop=10000)) + ".jpg",
                                        const_size_image, const_size_image)

                # Save the new face img
                l_np_faces.append(np_img_cut)

                # Creating new images non-face for training:
                for i in range(2):
                    #       0                 1             2       3        4
                    # <major_a xis_radius minor_axis_radius angle center_x center_y>

                    shift_x = randrange(np_img.shape[1] - int_feature[0]) + int_feature[0]
                    shift_y = randrange(np_img.shape[0] - int_feature[0]) + int_feature[0]

                    y_non_face = [shift_y - int_feature[0], shift_y + int_feature[0]]
                    x_non_face = [shift_x - int_feature[0], shift_x + int_feature[0]]

                    np_img_cut = basic_functs.cut_and_resize_img(x_non_face, y_non_face, np_img,
                                                              const_size_image)

                    # basic_functs.show_img(img_cut)
                    basic_functs.save_image(np_img_cut, "./train/non_face/n" +
                                            str(i) + "-" + str(randrange(1, stop=10000)) + ".jpg",
                                            const_size_image, const_size_image)

                    # Save the new non_face np_image
                    l_np_non_faces.append(np_img_cut)

        print "\n\n[INFO]: Read and Cutting Faces and Non-Faces Done"

        end = time.time()
        print "\tTime spend to organize: ", end - start, 'seconds', "\n"

        # Create a unique list with each np_image generate
        np_data_set = np.concatenate((l_np_faces, l_np_non_faces))
        np_data_set = np.asarray(np_data_set, dtype=np.uint8)

        # Create the labels
        np_label_set = np.concatenate(([1] * len(l_np_faces), [0] * len(l_np_faces) * 2))

        print "\tData_set size  (quant, size): ", np_data_set.shape
        print "\tLabel_set size (quant,):      ", np_label_set.shape, "\n"

        print "[INFO]: Creating the ArrayIterator of training set"

        # Create the array iterator with informations
        train_set = ArrayIterator(X=np_data_set,
                                  y=np_label_set, nclass=2, lshape=(3, const_size_image,
                                                                    const_size_image))
        end = time.time()
        print "\tTime spend to organize: ", end - start, 'seconds', "\n"

        return train_set, l_class_Figure
    """


def loading_set_for_testing(directories):
    """
    Procedure that loads the dataset for testing
    :param directories: List of directors
    :return: return the images.
    """
    print "[INFO]: Loading test FDDB"

    start = time.time()

    l_class_Figure = []

    # For each directory
    for directory in directories:

        # Opening the file
        f_paths = open(directory, "r")

        s_line = f_paths.readline()

        # For each s_path
        while s_line != "":
            # Create a new Figura Class
            l_class_Figure.append(Figura_Class())

            # Save the s_path
            l_class_Figure[-1].set_path(s_line[:len(s_line) - 1] + ".jpg")

            # Save the np_image
            l_class_Figure[-1].set_image(basic_functs.load_image("./data_sets/originalPics/" +
                                                                l_class_Figure[-1].get_path()))

            # If the np_image are black and white, delete it and jump to next
            if len(l_class_Figure[-1].get_image().shape) == 2:
                del l_class_Figure[-1]

            s_line = f_paths.readline()

        f_paths.close()

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return l_class_Figure


def making_regions(l_class_figure, batch_min_size, const_size_image):
    """
    Procedure that gets the figures test and create a lot of regions to do the tests
    :param l_class_figure: List of figures
    :param batch_size: Size of batch
    :param const_size_image: Size of each np_image to test
    :return: The regions list
    """
    print "[INFO]: Generating the Regions"

    start = time.time()
    l_batch_np_images = []

    # For each Image
    for Figure_original in l_class_figure:

        # Bound of each figure to test
        bound = const_size_image
        # Distance's interval of the face
        interval = bound - bound / 4
        l_new_images = []

        # Quantity of new images created
        int_count_new_images = 0

        #if "ig/img_648" in Figura_original.get_path():
        #    print Figura_original.get_image().shape

        # Verify if the region respect the bound of the face
        while (bound < Figure_original.get_image().shape[0] or bound < Figure_original.get_image().shape[1]):

            # Move on the Y axis
            for edge_y_axis in range(bound, Figure_original.get_image().shape[0], interval):

                # Move on the X axis
                for edge_x_axis in range(bound, Figure_original.get_image().shape[1],
                                         interval):

                    # Crop the new np_image
                    crop_np_img = Figure_original.get_image()[edge_y_axis - bound: edge_y_axis
                               ][:, edge_x_axis - bound: edge_x_axis][:]

                    # Verify if the np_image is little than the bound of algorithm
                    if bound > const_size_image:
                        l_new_images.append(basic_functs.resize_img(
                            crop_np_img, const_size_image))
                    else:
                        l_new_images.append(np.array(crop_np_img).reshape(-1))

                    # Add on the information of the region Figure list
                    Figure_original.l_faces_positions.append([edge_x_axis, edge_y_axis, bound])

                    int_count_new_images += 1


            # if the batch of this np_image list complete, stop to create new regions
            bound += 60
            interval = bound - bound / 4

        while len(l_new_images) < batch_min_size:
            np_img = Figure_original.get_image()

            shift_x = randrange(np_img.shape[1] - const_size_image / 2) + const_size_image
            shift_y = randrange(np_img.shape[0] - const_size_image / 2) + const_size_image

            y_non_face = [shift_y - const_size_image, shift_y + const_size_image]
            x_non_face = [shift_x - const_size_image, shift_x + const_size_image]

            np_img_cut = basic_functs.cut_and_resize_img(x_non_face, y_non_face, np_img, const_size_image)

            # Save the new non_face np_image
            l_new_images.append(np.array(np_img_cut).reshape(-1))

        l_batch_np_images.append(l_new_images)

    print "\tNumbers of Batches for cropped images:       ", len(l_batch_np_images)
    print "\t\tNumbers of cropped images of each np_image: ", len(l_batch_np_images[0])

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return np.array(l_batch_np_images)


def generate_inference(batches, batch_size, const_size_image):
    """
    Procedure that generates the inferences for test
    :param batches: list of batches of images
    :param batch_size: size of each batch
    :param const_size_image: size of images of algorithm
    :return: list of inference
    """

    print "[INFO]: Creating the ArrayIterators for test"

    start = time.time()

    l_inferences = []

    # For each batch from list
    count = 1
    for batch in batches:
        sys.stdout.write("\r\tProcessed: " + str(count) + " of " + str(len(batches)) +
                         ". \tCompleted: " +
                         str((count) / float(len(batches)) * 100.0 ) +
                         "%.\tTime spend until now: " +
                         str(time.time() - start) + "s")
        sys.stdout.flush()

        # Create a new list empty
        #x_new = np.zeros((batch_size, const_size_image *
        #                  const_size_image * 3), dtype=np.uint8)

        x_new = np.zeros((len(batch), const_size_image * const_size_image * 3), dtype=np.uint8)

        # Add the batch to the list
        x_new[0:len(batch)] = np.asarray(batch, dtype=np.uint8)

        # Create the array iterator from the new list
        inference_out = ArrayIterator(X=x_new, y=None, nclass=2,
                                      lshape=(3, const_size_image, const_size_image))

        # Save in a list
        l_inferences.append(inference_out)

        count += 1

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

    print "\tNumbers of Batches with ArrayIterators: ", len(l_inferences)

    # For each inference
    i = 1
    for inference in l_inferences:
        sys.stdout.write("\r\tProcessed: " + str(i) + " of " + str(len(l_inferences)) +
                         ". \tCompleted: " + str(i / float(len(l_inferences)) * 100.0) +
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
def mark_result(np_img, position):
    """
    Procedure that mark the result of process
    :param image: np_image to mark
    :param position: position to mark
    :param s_path: s_path to save
    :return: do not have return
    """

    np_img_marked = np_img

    #np_img_marked = np.array(image)
    coords = position

    edge_x_axis = coords[1]
    edge_y_axis = coords[0]
    bound = coords[2]

    np_img_marked.flags.writeable = True

    # mark the y axis
    value = 255
    for y in range(edge_y_axis - bound, edge_y_axis):
        np_img_marked[edge_x_axis, y, 0] = value
        np_img_marked[edge_x_axis - bound, y, 0] = value
        np_img_marked[edge_x_axis, y+1, 0] = value
        np_img_marked[edge_x_axis - bound, y+1, 0] = value

        np_img_marked[edge_x_axis, y, 1:2] = 0
        np_img_marked[edge_x_axis - bound, y, 1:2] = 0

    # Mark the x axis
    for x in range(edge_x_axis - bound, edge_x_axis):
        np_img_marked[x, edge_y_axis, 0] = value
        np_img_marked[x, edge_y_axis - bound, 0] = value
        np_img_marked[x+1, edge_y_axis, 0] = value
        np_img_marked[x+1, edge_y_axis - bound, 0] = value

        np_img_marked[x, edge_y_axis, 1:2] = 0
        np_img_marked[x, edge_y_axis - bound, 1:2] = 0

    return np_img_marked


def analyze_results(l_out, l_Figura_class):
    print "[INFO]: Analyzing the Results"

    index = 0
    start = time.time()

    if os.path.exists("./out/"):
        shutil.rmtree("./out/")

    os.mkdir("./out/")
    os.mkdir("./out/f/")
    os.mkdir("./out/n/")

    # Para cada batch de images 
    for batch in l_out:

        # para cada regiao
        num_region = 0
        image_marked = False
        for region in batch:
            #sys.stdout.write("\r\tProcessed: " + str(index + 1) + ":" + str(num_region+1) + " of " + str(len(l_out)) +
            #                 ". \tCompleted: " + str(index + 1 / float(len(l_out)) * 100.0) +
            #                 "%.\tTime spend until now: " + str(time.time() - start) + "s")
            #sys.stdout.flush()
            # se for encontrado algum rosto

            if region[0] > 0.5:
                print "Face Detectada: ", region
                image_marked = True

                np_img = l_Figura_class[index].get_image()


                np_img_cut = mark_result(np_img,
                               l_Figura_class[index].l_faces_positions[num_region])

                l_Figura_class[index].set_image(np_img_cut)

            if image_marked:
                basic_functs.save_image(l_Figura_class[index].get_image(),
                                        "./out/f/f" + str(index) + "-" +
                                        str(num_region) + ".jpg")

            num_region += 1

            if num_region >= len(l_Figura_class[index].l_faces_positions):
                break


        index += 1

    end = time.time()
    print "\n\tTime spend to generate the outputs: ", end - start, 'seconds', "\n"

"""
#classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
 "horse", "ship", "truck"]
#print classes
#classes = ["No", "Yes"]

out = model.get_outputs(inference_set)

print out[0]
#print classes[out[0].argmax()]

"""