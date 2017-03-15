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
