# Search files on path
import os.path
#
import sys

import cPickle

# Backend
from neon.backends import gen_backend

from neon.data import CIFAR10, ArrayIterator
from neon.layers import Conv, Affine, Pooling, GeneralizedCost
from neon.initializers import Uniform
from neon.transforms.activation import Rectlin, Softmax
from neon.transforms import CrossEntropyMulti, Misclassification
from neon.models import Model
from neon.optimizers import GradientDescentMomentum, RMSProp
from neon.callbacks.callbacks import Callbacks

from PIL import Image

# download the image
import urllib
from random import randrange
# crop and resize to 32x32
from PIL import Image
import time
import numpy as np
import basic_functs
from neon.initializers import Gaussian


class Figura_Class:
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



def build_model(network_depth, num_features):

    # Layers
    # Networks layers
    # As Neon supports many layers types, including Linear Convolution,
    #    Bias, Activation, and Pooling. For that, neon provides shortcuts:
    #    Conv = Convolution + Bias + Activation
    #    Affine = Linear + Bias + Activation

    print "[INFO]: Making the layers."

    # We are going to create a tiny network with two Conv, two Pooling, and two Affine layers.
    #    fshape = (width, height, # of filters)

    init_uni = Uniform(low=-0.1, high=0.1)
    layers = [Conv(fshape=(5, 5, 16),
                   init=init_uni,
                   activation=Rectlin()),
              Pooling(fshape=2, strides=2),
              Conv(fshape=(5, 5, 32),
                   init=init_uni,
                   activation=Rectlin()),
              Pooling(fshape=2, strides=2),
              Affine(nout=500,
                     init=init_uni,
                     activation=Rectlin()),
              Affine(nout=2,
                     init=init_uni,
                     activation=Softmax())]

    """
    init_norm = Gaussian(loc=0.0, scale=0.01)
    layers = [Affine(nout=100, init=init_norm, activation=Rectlin()),
              (Affine(nout=2, init=init_norm, activation=Softmax()))]
    """

    print "[SETU]: Setting the layers up."
    # Seting up of the model
    model = Model(layers)

    return model



def loading_set_for_training(diretorios):
    print "[INFO]: Loading FDDB"

    start = time.time()
    l_figure_class = []
    number_figures = 0

    for diretorio in diretorios:
        # Lendo diretorio das imagens
        file_features = open(diretorio, "r")

        # criando uma lista de diretorios
        s_line = file_features.readline()

        while s_line != "":

            # Instancia uma nova classe Figura
            l_figure_class.append(Figura_Class())

            # print "Linha lida: ", s_line

            # Recebe o caminho da imagem
            l_figure_class[number_figures].set_path(s_line[:len(s_line) - 1] + ".jpg")

            l_figure_class[number_figures].set_image(
                basic_functs.load_image("./data_sets/originalPics/" +
                                        l_figure_class[number_figures].get_path()))

            if len(l_figure_class[number_figures].get_image().shape) == 3:

                # basic_functs.show_img(l_figure_class[-1].get_image())
                # sys.exit(11)

                # print "Dimensoes Imagem", l_figure_class[number_figures].get_image().shape

                # Define quantidade de rostos
                s_line = file_features.readline()

                # print "Linha lida: ", s_line
                l_figure_class[number_figures].set_number_faces(int(s_line))

                number_faces = l_figure_class[number_figures].get_number_faces()

                # Le as informacoes de cada rosto
                for index_face in range(0, number_faces):
                    s_line = file_features.readline()

                    # print "Linha lida: ", s_line

                    # realiza o split
                    # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
                    l_features_of_one_figure_buffer = s_line.split(" ")

                    # Elimina o " " e o "1"
                    l_features_of_one_figure_buffer_cut = \
                        l_features_of_one_figure_buffer[:len(l_features_of_one_figure_buffer) - 2]

                    # Adiciona a nova face no objeto
                    l_figure_class[number_figures].add_face_position(l_features_of_one_figure_buffer_cut)

                number_figures += 1

                s_line = file_features.readline()

            else:
                del l_figure_class[number_figures]

                s_line = file_features.readline()

                for i in range(int(s_line) + 1):
                    s_line = file_features.readline()
                    # print "Linha lida: ", s_line

        l_faces = []
        l_non_faces = []

        file_features.close()


    print "[INFO]: Cutting the images, getting the faces and non-faces"

    for i in range(0, number_figures):
        sys.stdout.write("\rProcessed: " + str(i) + " of " + str(number_figures) + ". \tCompleted: " + str((i+1) / float(number_figures) * 100.0 ) + "%")
        sys.stdout.flush()
        #       0                 1             2       3        4
        # <major_axis_radius minor_axis_radius angle center_x center_y>

        # Load the informations
        features = l_figure_class[i].get_face_positions()
        img = l_figure_class[i].get_image()

        for feature in features:

            feature_int = [int(float(x)) for x in feature]

            if feature_int[4] + feature_int[0] > img.shape[0]:
                feature_int[4] -= feature_int[4] + feature_int[0] - (img.shape[0] - 1)

            if feature_int[3] + feature_int[0] > img.shape[1]:
                # feature_int[3] -= feature_int[3] - (img.shape[1] + 1)
                feature_int[3] -= feature_int[3] + feature_int[0] - (img.shape[1] - 1)

            # Do de crop
            y = [feature_int[4] - feature_int[0], feature_int[4] + feature_int[0]]
            x = [feature_int[3] - feature_int[0], feature_int[3] + feature_int[0]]

            img_cut = basic_functs.cut_and_resize_img(x, y, img)

            # basic_functs.show_img(img_cut)

            # Save the new face img
            l_faces.append(img_cut)

            # NON FACE:

            #       0                 1             2       3        4
            # <major_axis_radius minor_axis_radius angle center_x center_y>

            shift_x = randrange(img.shape[1] - feature_int[0])
            shift_y = randrange(img.shape[0] - feature_int[0])


            y_non_face = [shift_y - feature_int[0], shift_y + feature_int[0]]
            x_non_face = [shift_x - feature_int[0], shift_x + feature_int[0]]

            img_cut = basic_functs.cut_and_resize_img(x_non_face, y_non_face, img)

            # basic_functs.show_img(img_cut)

            # Save the new non_face image
            l_non_faces.append(img_cut)

    print "\n[INFO]: Read and Cutting Faces and Non-Faces Done"

    np_data_set = np.concatenate((l_faces, l_non_faces))
    np_label_set = np.concatenate(([1] * len(l_faces), [0] * len(l_faces)))

    print "\tData_set size (quant, size): ", np_data_set.shape
    print "\tLabel_set size (quant,):     ", np_label_set.shape

    print "[INFO]: Creating the ArrayIterator of training set"

    train_set = ArrayIterator(X=np_data_set, y=np_label_set, nclass=2, lshape=(3, 120, 120))

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    return train_set


def loading_set_for_testing(paths):
    print "[INFO]: Loading FDDB"

    start = time.time()

    # Lendo diretorio das imagens
    f_paths = open(paths, "r")

    # criando uma lista de diretorios
    s_line = f_paths.readline()

    l_figure_class = []

    while s_line != "":
        # Instancia uma nova classe Figura
        l_figure_class.append(Figura_Class())

        # print "Linha lida: ", s_line

        # Recebe o caminho da imagem
        l_figure_class[-1].set_path(s_line[:len(s_line) - 1] + ".jpg")

        l_figure_class[-1].set_image(basic_functs.load_image("./data_sets/originalPics/" +
                                                            l_figure_class[-1].get_path()))

        s_line = f_paths.readline()


    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    return l_figure_class


def making_regions(l_Figure_class, batch_size):
    print "[INFO]: Generating the Regions"

    start = time.time()

    l_batch = []

    # For each Image
    for Figura_original in l_Figure_class:

        bound = 120
        interval = 90
        l_new_images = []

        quant_new_images = 0

        while bound < Figura_original.get_image().shape[0] or \
                        bound < Figura_original.get_image().shape[1]:

            # Y
            for edge_y_axis in range(bound, Figura_original.get_image().shape[0], interval):

                # X
                for edge_x_axis in range(bound, Figura_original.get_image().shape[1], interval):
                    crop_img = Figura_original.get_image()[edge_y_axis - bound: edge_y_axis
                               ][:, edge_x_axis - bound: edge_x_axis][:]

                    #print edge_y_axis, edge_x_axis, crop_img.shape, Figura_original.get_image().shape

                    # TODO ADICIONAR A NOVA IMAGEM NA LIST
                    if bound > 120:
                        l_new_images.append(basic_functs.resize_img(crop_img))
                    else:
                        l_new_images.append(np.array(crop_img).reshape(-1))

                    quant_new_images += 1

                    if quant_new_images >= batch_size:
                        break

                if quant_new_images >= batch_size:
                    break

            if quant_new_images >= batch_size:
                break

            bound += 60
            interval = bound - bound / 4

        l_batch.append(l_new_images)


    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    return np.array(l_batch)


def test_inference(batches, model, batch_size):

    print "[INFO]: Making the tests"

    start = time.time()

    l_inferences = []
    l_out = []

    #batch = batches[0]
    for batch in batches:

        x_new = np.zeros((batch_size, 120 * 120 * 3))

        x_new[0:len(batch)] = batch

        inference_out = ArrayIterator(X=x_new, y=None, nclass=2, lshape=(3, 120, 120))

        l_inferences.append(inference_out)


    for inference in l_inferences:
        out = model.get_outputs(inference)

        l_out.append(out)


    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    print "[INFO]: Tests Done"

    return l_out


def analyze_results(l_out, l_Figura_class):
    print "[INFO]: Analyzing the Results"

    for batch in l_out:
        for region in batch:
            print region

            sys.exit(99)

"""

# an image of a frog from wikipedia
#img_source = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/55/Atelopus_zeteki1.jpg/440px-Atelopus_zeteki1.jpg"
img_source = "http://www.blog2.it/wp-content/uploads/2010/12/labrador-retriever.jpeg"

urllib.urlretrieve(img_source, filename="image.jpg")

img_source = "image.jpg"
#img_source = "human1.jpg"

img = Image.open(img_source)
crop = img.crop((0, 0, min(img.size), min(img.size)))
crop.thumbnail((32, 32))
crop = np.asarray(crop, dtype=np.float32)

x_new = np.zeros((128, 3072), dtype=np.float32)
x_new[0] = crop.reshape(1, 3072) / 255

inference_set = ArrayIterator(x_new, None, nclass=number_class, lshape=(3, 32, 32))

#classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#print classes
#classes = ["No", "Yes"]

out = model.get_outputs(inference_set)

print out[0]
#print classes[out[0].argmax()]

"""