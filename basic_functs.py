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


def show_img(img):

    img_np = np.array(img)

    if len(img_np.shape) == 1:
        img_reshaped = img_np.reshape((120, 120, 3))
    else:
        img_reshaped = img_np

    img_out = Image.fromarray(np.asarray(img_reshaped, dtype="uint8"), "RGB")
    img_out.show()


def cut_and_resize_img(x, y, img_original):
    img = img_original
    standard = hsize = 120

    x_range = range(x[0], x[1])
    y_range = range(y[0], y[1])

    # img[y_range][:, x_range]
    matrix_cropped = img[y_range][:, x_range][:]

    #np_cropped = np.array(matrix_cropped)
    img_thumbnail = Image.fromarray(np.asarray(matrix_cropped, dtype="uint8"), "RGB")

    img_thumbnail = img_thumbnail.resize((standard, hsize), Image.ANTIALIAS)

    return np.array(img_thumbnail).reshape(-1)


def resize_img(img):
    standard = 120

    new_img = Image.fromarray(img, 'RGB')

    crop_img = new_img.resize((standard, standard), Image.ANTIALIAS)

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
