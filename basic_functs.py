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
import numpy as np
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


def cut_img(x, y, img):
    standard = 120

    x_range = range(x[0], x[1])
    y_range = range(y[0], y[1])

    new_img = Image.fromarray(img[y_range][:, x_range][:], 'RGB')

    # wpercent = (standard / float(new_img.size[0]))
    # hsize = int((float(new_img.size[1]) * float(wpercent)))
    hsize = 120

    crop_img = new_img.resize((standard, hsize), Image.ANTIALIAS)

    return np.array(crop_img).reshape(-1)


def unpickle(directory):
    fo = open(directory, "rb")
    dict_loading = cPickle.load(fo)
    fo.close()
    return dict_loading


def load_image(directory):
    img_loading = Image.open(directory)
    img_loading.load()
    return np.asarray(img_loading, dtype="int32")
    # return img_loading


def save_image(np_data, directory):
    img_saving = Image.fromarray(np.asarray(np.clip(np_data, 0, 255), dtype="uint8"), "L")
    img_saving.save(directory)


def loading_set_for_training(path_features):
    print "[INFO]: Loading FDDB"
    # Lendo diretorio das imagens
    file_features = open(path_features, "r")

    l_figure_class = []
    number_figures = 0

    # criando uma lista de diretorios
    s_line = file_features.readline()

    while s_line != "":
        # Instancia uma nova classe Figura
        l_figure_class.append(Figura_Class())

        # print "Linha lida: ", s_line

        # Recebe o caminho da imagem
        l_figure_class[number_figures].set_path(s_line[:len(s_line) - 1] + ".jpg")

        l_figure_class[number_figures].set_image(load_image("./data_sets/originalPics/" +
                                                            l_figure_class[number_figures].get_path()))

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

    l_faces = []
    l_non_faces = []

    print "[INFO]: Cutting the images, getting the faces and non-faces"

    for i in range(0, number_figures):
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

            img_cut = cut_img(x, y, img)

            # Save the new face img
            l_faces.append(img_cut)

            # NON FACE:

            shift_x = randrange(img.shape[1] - feature_int[0])
            shift_y = randrange(img.shape[0] - feature_int[0])

            y = [shift_y - feature_int[0], shift_y + feature_int[0]]
            x = [shift_x - feature_int[0], shift_x + feature_int[0]]

            img_cut = cut_img(x, y, img)

            # Save the new non_face image
            l_non_faces.append(img_cut)

    print "[INFO]: Read and Cutting Faces and Non-Faces Done"

    np_data_set = np.concatenate((l_faces, l_non_faces))
    np_label_set = np.concatenate(([1] * len(l_faces), [0] * len(l_faces)))

    print "\t Data_set size: ", np_data_set.shape
    print "\t Label_set size:", np_label_set.shape

    print "[INFO]: Creating the ArrayIterator of training set"

    train_set = ArrayIterator(X=np_data_set, y=np_label_set, nclass=2, lshape=(3, 120, 120))

    return train_set


def verify_args():
    # if len(sys.argv) == 2:
    if True:
        # do_again = int(sys.argv[1])
        train_again = 1
        if train_again != 0 and train_again != 1:
            print "[ERRO]: Invalid Parameters."
            print "[INFO]: Use name_of_program.py option_training."
            print "[INFO]: option_training: 1 - create a new Neural Network; 0 - Use already saved."
            sys.exit()
    else:
        print "[ERRO]: Invalid Parameters."
        print "[INFO]: Use name_of_program.py option_training."
        print "[INFO]: option_training: 1 - create a new Neural Network; 0 - Use already saved."
        sys.exit()

    return train_again
