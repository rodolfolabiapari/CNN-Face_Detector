
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

# download the image
import urllib
# crop and resize to 32x32
from PIL import Image
import numpy as np
from neon.initializers import Gaussian

import time


# an image of a frog from wikipedia
# img_source =
# "https://upload.wikimedia.org/wikipedia/
# commons/thumb/5/55/Atelopus_zeteki1.jpg/440px-Atelopus_zeteki1.jpg"
# img_source = "https://www.scienceabc.com/wp-content/uploads/2016/05/horse-running.jpg"

# urllib.urlretrieve(img_source, filename="image.jpg")

model = Model("pdi_model.prm")

img = Image.open("human2.jpg")
crop = img.crop((0, 0, min(img.size), min(img.size)))
crop.thumbnail((32, 32))
crop = np.asarray(crop, dtype=np.float32)

x_new = np.zeros((128, 3072), dtype=np.float32)
x_new[0] = crop.reshape(1, 3072) / 255

inference_set = ArrayIterator(x_new, None, nclass=2, lshape=(3, 32, 32))

# classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
classes = ["No", "Yes"]


out = model.get_outputs(inference_set)


print out[0]
print classes[out[0].argmax()]

print ("\t[FINS]\n")
