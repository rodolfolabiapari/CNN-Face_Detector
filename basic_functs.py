# Search files on path
import os.path
#
import sys
import cPickle

# crop and resize to 32x32
from PIL import Image
import numpy as np


def show_img(np_img, size_image):
    """
    Function that show the np_image without save it.
    :param img: np_image that will backend showed
    :param size_image: size of np_image
    :return: do not have a return
    """

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
        img_reshaped = np_img.reshape((size_image, size_image, 3))
    else:
        img_reshaped = np_img

    # Convert the numpy to a Image
    img_out = Image.fromarray(np.asarray(img_reshaped, gfh gf fhg dtype="uint16"))

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
    x_range = range(x[0], x[1])
    y_range = range(y[0], y[1])

    # Cut the np_image
    matrix_cropped = img[y_range][:, x_range][:]

    # Transform to a Image Object
    img_thumbnail = Image.fromarray(np.asarray(matrix_cropped), "RGB")

    # Resize the np_image
    img_thumbnail = img_thumbnail.resize((size_image, size_image), Image.ANTIALIAS)

    # Return the new np_image
    return np.array(img_thumbnail, dtype="uint16").reshape(-1)


def resize_img(img, size_image):
    """
    Receive a np_image from parameter and resize it to the CONST_size_image_algorithm
    :param img: Image
    :param size_image: new format
    :return:
    """

    # Convert to a Image Object
    new_img = Image.fromarray(img, 'RGB')

    # Resize it
    crop_img = new_img.resize((size_image, size_image), Image.ANTIALIAS)

    # Return the new np_image
    return np.array(crop_img, dtype="uint16").reshape(-1)


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
    img_loading = Image.open(directory)
    img_loading.load()
    np_img = np.asarray(img_loading, dtype="uint16")
    return np_img


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

        img_reshaped = np_img.reshape((lin, col, 3))

    else:
        img_reshaped = np_img


    # Save the np_image
    img_out = Image.fromarray(np.asarray(img_reshaped, dtype="uint16"), "RGB")
    img_out.save(directory)

    sys.exit(22)


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
