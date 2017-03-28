
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

import time
import numpy as np  # Work with vectors and matrices
import basic_functs
import os
import shutil
from neon.layers import Conv, Convolution, Dropout, Affine, Activation, Pooling, GeneralizedCost
from neon.initializers import Uniform, Gaussian
from neon.transforms.activation import Rectlin, Softmax
from neon.transforms import CrossEntropyMulti, CrossEntropyBinary, Misclassification
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp, Schedule
from neon.callbacks.callbacks import Callbacks
import os.path
#
import sys
from random import randrange
from neon.data import ArrayIterator


def train_model_fddb_94(train_set, test_set, num_epochs, learning_rate, momentum):
    print "[DOWN]: Getting the dataset of training."

    print "[INFO]: The Neural Network was not created yet."
    print "[INFO]: Starting procedure of creating and training."

    # Procedure of Costs
    #     Setting up the cost function of network output

    print "[INFO]: Creating the cost function."
    cost = GeneralizedCost(costfunc=CrossEntropyMulti())

    # Procedure for Build Model

    print "[INFO]: Making the layers."

    # Network with two Conv, two Pooling, and two Affine layers.

    print "\tCreating a Convolutional Network"
    init_uni = Uniform(low=-0.1, high=0.1)

    print "\t\tConv(   fshape=(5, 5, 4)"
    print "\t\tPooling(fshape=(2, 2)"
    print "\t\tConv(   fshape=(3, 3, 14)"
    print "\t\tPooling(fshape=(2, 2)"
    print "\t\tAffine( nout=14"
    print "\t\tAffine( nout=2\n"

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (34-5+1 , 34-5+1) = (30, 30)
    # maxpooling reduces this further to (30/2, 30/2) = (15, 15)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 15, 15)

    layers = [  # Conv(fshape=(5, 5, 4), init=init_uni, activation=Rectlin()),
        Convolution(fshape=(5, 5, 4), init=init_uni),

        Pooling(fshape=(2, 2)),

        # Conv(fshape=(3, 3, 14), init=init_uni, activation=Rectlin()),
        Convolution(fshape=(3, 3, 14), init=init_uni),

        Pooling(fshape=(2, 2)),

        Affine(nout=14, init=init_uni, activation=Rectlin(), batch_norm=True),

        Affine(nout=2, init=init_uni, activation=Softmax())]

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
    callbacks = Callbacks(model, test_set)

    # Training

    print "[INFO]: Making the Neural Network with", num_epochs, "epochs."

    start = time.time()
    model.fit(dataset=train_set, cost=cost, optimizer=optimizer,
              num_epochs=num_epochs, callbacks=callbacks)
    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'
    print "\tTime spend to organize: ", (end - start) / 60.0, 'minutes\n'

    return model



def do_validation(valid_set, model):
    print "----------------------------------------------" \
          "\n[INFO]: Starting the procedure of tests."

    # Calcule the Miss classification error by framework
    miss_test = False  # or True
    if miss_test:
        print "[INFO]: Checking the Miss classification of error."
        start = time.time()
        error_pct = 100 * model.eval(valid_set[2], metric=Misclassification())
        end = time.time()
        print "\tTime spend to organize:   ", end - start, 'seconds'
        print "\tMiss classification error: %.3f%%" % error_pct, "\n"

    # Test each one batch from list of batches
    l_out = test_inference(valid_set, model)

    return l_out


def loading_set_for_training(num_files_94, l_directories_fddb_train, const_size_image):
    """
    Procedure to read dataset for training.
    :param directorys: directorys for read the images. It can backend more than one.
    :param const_size_image: Size of images to train
    :return: A lot of things
    """

    l_np_faces_fddb = []
    l_np_faces_94 = []

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
    l_class_figure_94, l_np_faces_94 = basic_functs.load_94_and_generate_faces(num_files_94, const_size_image)

    print "\tNumber of Faces:     ", len(l_np_faces_94), "\n"
    # FDDB

    print "[INFO]: Loading training FDDB data-set"
    l_class_figure_fddb = basic_functs.load_fddb(l_directories_fddb_train, const_size_image)

    l_np_non_faces_fddb = basic_functs.generate_non_faces_fddb(l_class_figure_fddb, const_size_image)

    if len(l_np_faces_fddb) == 0:
        l_np_faces = l_np_faces_94
    elif len(l_np_faces_94) == 0:
        l_np_faces = l_np_faces_fddb
    else:
        l_np_faces = np.concatenate((l_np_faces_fddb, l_np_faces_94))

    print "\tNumber of Faces:     ", len(l_np_faces_fddb)

    print "\tNumber of Non-Faces: ", len(l_np_non_faces_fddb), "\n"

    l_np_non_faces = l_np_non_faces_fddb

    print "[INFO]: Images read successfully"
    print "\tNumber of Faces:     ", len(l_np_faces)
    print "\tNumber of Non-Faces: ", len(l_np_non_faces)

    arrayIterator_train_set = basic_functs.making_arrayIterator(l_np_faces, l_np_non_faces, const_size_image)

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    return arrayIterator_train_set



def loading_set_for_validation(list_images, batch_min_size, const_size_image):
    """
    Procedure to read dataset for training.
    :param directorys: directorys for read the images. It can backend more than one.
    :param const_size_image: Size of images to train
    :return: A lot of things
    """

    # Calcule the time
    start = time.time()

    # FDDB

    print "[INFO]: Loading FDDB data-set"
    valid_figures = basic_functs.load_fddb([list_images], const_size_image)

    # Create a list of batches for test
    l_batches_test = making_regions(valid_figures, batch_min_size, const_size_image)

    # Generate the inference lists for tests
    valid_set = generate_inference(l_batches_test, batch_min_size, const_size_image)

    return valid_set, valid_figures


def loading_set_for_testing(directories_test, batch_min_size, const_size_image):
    # Create a list of figures from test dataset
    test_figures = loading_testing_data_set(directories_test)

    # Create a list of batches for test
    l_batches_test = making_regions(test_figures, batch_min_size, const_size_image)

    # Generate the inference lists for tests
    test_set = generate_inference(l_batches_test, batch_min_size, const_size_image)

    return test_set, test_figures


def loading_testing_data_set(directories):
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
            l_class_Figure[-1].set_image(
                basic_functs.load_image("./data_sets/originalPics/" + l_class_Figure[-1].get_path()))

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

    factor = 4
    i = 0

    # For each Image
    for Figure_original in l_class_figure:
        sys.stdout.write("\r\tProcessed: " + str(i + 1) + " of " +
                         str(len(l_class_figure)) +
                         ". \tCompleted: " +
                         str((i + 1) / float(len(l_class_figure)) * 100.0) + "%")
        sys.stdout.flush()

        # Bound of each figure to test
        bound = const_size_image
        # Distance's interval of the face
        interval = bound / factor
        l_new_images = []

        # Quantity of new images created
        int_count_new_images = 0

        # if "ig/img_648" in Figura_original.get_path():
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
                                  ][:, edge_x_axis - bound: edge_x_axis]

                    # basic_functs.save_image(mark_result(Figure_original.get_image(), [edge_x_axis, edge_y_axis, bound]), "./regions/" + str(time.time()) + ".jpg")

                    # Verify if the np_image is little than the bound of algorithm
                    if bound > const_size_image:
                        l_new_images.append(basic_functs.resize_img(
                            crop_np_img, const_size_image))
                    else:
                        l_new_images.append(np.array(crop_np_img).reshape(-1))

                    # Add on the information of the region Figure list
                    Figure_original.l_faces_positions_regions.append([edge_x_axis, edge_y_axis, bound])

                    int_count_new_images += 1

            # if the batch of this np_image list complete, stop to create new regions
            bound += 60
            interval = bound / factor

        while len(l_new_images) < batch_min_size:
            np_img = Figure_original.get_image()

            shift_x = randrange(np_img.shape[1] - const_size_image / 2) + const_size_image
            shift_y = randrange(np_img.shape[0] - const_size_image / 2) + const_size_image

            y_non_face = [shift_y - const_size_image, shift_y + const_size_image]
            x_non_face = [shift_x - const_size_image, shift_x + const_size_image]

            np_img_cut = basic_functs.cut_and_resize_img(x_non_face, y_non_face, np_img, const_size_image)

            # Save the new non_face np_image
            l_new_images.append(np.array(np_img_cut).reshape(-1))

            # Add on the information of the region Figure list
            Figure_original.l_faces_positions_regions.append([shift_x, shift_y, bound])

        l_batch_np_images.append(l_new_images)

    print "\tNumbers of Batches for cropped images:          ", len(l_batch_np_images)
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
                         str((count) / float(len(batches)) * 100.0) +
                         "%.\tTime spend until now: " +
                         str(time.time() - start) + "s")
        sys.stdout.flush()

        # Create a new list empty
        # x_new = np.zeros((batch_size, const_size_image *
        #                  const_size_image * 3), dtype=np.uint8)

        x_new = np.zeros((len(batch), const_size_image * const_size_image), dtype=np.uint8)

        # Add the batch to the list
        x_new[0:len(batch)] = np.asarray(batch, dtype=np.uint8)

        # Create the array iterator from the new list
        inference_out = ArrayIterator(X=x_new, y=None, nclass=2, lshape=(1, const_size_image, const_size_image))

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

