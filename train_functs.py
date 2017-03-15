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