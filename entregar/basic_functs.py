
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
import os
import shutil
import matplotlib.pyplot as plt

import math
import os.path
#
import sys
import cPickle

# crop and resize to 32x32
from PIL import Image
import numpy as np
from random import randrange, shuffle
from neon.data import ArrayIterator

def save_results(l_out):
    print "[INFO]: Saving the Output"
    f = open("./out/l_out.txt", "w+")

    i = 0
    for batch in l_out:
        f.write(str(i) + "\n")
        for region in batch:
            f.write("\t [ {:.4e}".format(region[0]))
            if (region[0] > 0.5):
                f.write(" >>>>> ")
            else:
                f.write("       ")
            f.write("{:.4e} ]".format(region[1]))
            f.write("\n")

        f.write("\n")

        i += 1

    f.close()


def load_image(directory):
    """
    Function to read a unique np_image from directory
    :param directory: Directory to np_image
    :return: np_image in format numpy
    """
    img_loading = Image.open(directory).convert('L')
    img_loading.load()

    return np.asarray(img_loading, dtype=np.uint8)


def save_image(np_img, directory, lin=None, col=None):
    """
    Save a np_image in a specific directory
    :param img: the np_image
    :param directory: the directory
    :param lin: rows of np_image
    :param col: columns of np_image
    """

    # Verify if it np_image is on the correct shape
    if len(np_img.shape) == 1:
        if lin == None:
            print "ERROR, Defina o valor lin ou col"
            sys.exit(88)

        img_reshaped = np_img.reshape((lin, col))

    else:
        img_reshaped = np_img

    # Save the np_image
    img_out = Image.fromarray(np.asarray(img_reshaped, dtype=np.uint8), "L")

    img_out.save(directory)


def making_arrayIterator(l_np_faces, l_np_non_faces, const_size_image):

    start = time.time()

    # Create a unique list with each np_image generate
    np_data_set = np.concatenate((l_np_faces, l_np_non_faces))
    np_data_set = np.asarray(np_data_set, dtype=np.uint8)

    # Create the labels
    np_label_set = np.concatenate(([0] * len(l_np_faces), [1] * len(l_np_non_faces)))

    print "\n\tData_set size  (quant, size): ", np_data_set.shape
    print "\tLabel_set size (quant,):      ", np_label_set.shape, "\n"

    global NUM_IMAGES_TRAINED
    NUM_IMAGES_TRAINED = np_label_set.shape[0]

    print "[INFO]: Creating the ArrayIterator of training set"

    # Create the array iterator with information
    train_set = ArrayIterator(X=np_data_set, y=np_label_set, nclass=2, lshape=(1, const_size_image, const_size_image))

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return train_set


def load_fddb(directories, size_image):

    l_class_Figure = []

    start = time.time()

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
            l_class_Figure[-1].set_path(s_line[:len(s_line) - 1] + ".jpg")

            np_image = load_image("./data_sets/originalPics/" +
                                        l_class_Figure[-1].get_path())

            # Load the np_image
            l_class_Figure[-1].set_image(np_image)

            # show_img(l_class_Figure[-1].get_image())

            # Read quantity of faces
            s_line = file_features.readline()

            int_number_faces = int(s_line)

            l_class_Figure[-1].set_number_faces(int_number_faces)

            # Read the informations of each face
            for index_face in range(0, int_number_faces):
                # Read the line and cut everthing it is not necessary
                s_line = file_features.readline()

                # <major_axis_radius minor_axis_radius angle center_x center_y 1>.
                features_one_figure = s_line.split(" ")

                features_one_figure_cut = \
                    features_one_figure[:len(features_one_figure) - 2]

                # Add the new face in object
                l_class_Figure[-1].add_face_position_original(
                    features_one_figure_cut)

            s_line = file_features.readline()

        # Close the actual file
        file_features.close()

    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return l_class_Figure


def generate_faces_fddb(l_class_Figure, size_image):

    # Create a list of faces and non_faces
    l_faces = []

    # Start tue cutting

    print "[INFO]: Cutting the images, getting the faces"

    start = time.time()

    # For each np_image
    for i in range(0, len(l_class_Figure)):
        # Printe some informations
        sys.stdout.write("\r\tProcessed: " + str(i + 1) + " of " + str(len(l_class_Figure)) + ". \tCompleted: " +
                         str((i + 1) / float(len(l_class_Figure)) * 100.0) + "%")

        sys.stdout.flush()

        #       0                 1             2       3        4
        # <major_axis_radius minor_axis_radius angle center_x center_y>

        # Load the information from this np_image i
        features = l_class_Figure[i].get_face_positions()
        np_img = l_class_Figure[i].get_image()

        j = 0
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
            np_img_cut = cut_and_resize_img(x, y, np_img, size_image)

            # show_img(np_img_cut, const_size_image)

            save_image(np_img_cut, "./train/face/z" + str(i) + "-" +
                                    str(j) + ".jpg",
                                    size_image, size_image)

            # Save the new face img
            l_faces.append(np_img_cut)

            j += 1


    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds', "\n"

    return l_faces


def generate_non_faces_fddb(l_class_Figure, size_image):

    l_np_non_faces = []

    # Start tue cutting

    print "[INFO]: Cutting the images, getting the non-faces"

    start = time.time()

    # For each np_image
    for i in range(0, len(l_class_Figure)):

        # Print some informations
        sys.stdout.write("\r\tProcessed: " + str(i + 1) + " of " +
                         str(len(l_class_Figure)) +
                         ". \tCompleted: " +
                         str((i + 1) / float(len(l_class_Figure)) * 100.0) + "%")
        sys.stdout.flush()

        features = l_class_Figure[i].get_face_positions_original()

        int_features = []

        # Convert the string to int
        for feat in features:
            int_features.append([int(float(x)) for x in feat])

        np_img = l_class_Figure[i].get_image()

        if np_img.shape[1] - int_features[0][0] > int_features[0][0] + 90 and np_img.shape[0] - int_features[0][0] > \
                        int_features[0][0] + 90 and len(int_features) < 3:

            # for each face in the np_image
            for generations in range(0, 20):

                flag_igual_face_x = True
                flag_igual_face_y = True

                while flag_igual_face_x or flag_igual_face_y:
                    # shift_x = randrange(np_img.shape[1] - size_image / 2) + size_image
                    # shift_y = randrange(np_img.shape[0] - size_image / 2) + size_image

                    if flag_igual_face_x:
                        shift_x = randrange(np_img.shape[1] - size_image / 2) + size_image / 2
                        flag_igual_face_x = False

                    if flag_igual_face_y:
                        shift_y = randrange(np_img.shape[0] - size_image / 2) + size_image / 2
                        flag_igual_face_y = False

                    for int_feat in int_features:
                        if not flag_igual_face_x and abs(shift_x - int_feat[3]) < int_feat[0] * 0.7:
                            flag_igual_face_x = True

                        if not flag_igual_face_y and abs(shift_y - int_feat[4]) < int_feat[0] * 0.7:
                            flag_igual_face_y = True

                y_non_face = [shift_y - size_image, shift_y + size_image]
                x_non_face = [shift_x - size_image, shift_x + size_image]

                np_img_cut = cut_and_resize_img(x_non_face, y_non_face, np_img, size_image)


                # show_img(img_cut)
                save_image(np_img_cut, "./train/non_face/z" + str(i) + "-" + str(generations) + ".jpg", size_image,
                           size_image)

                # Save the new non_face np_image
                l_np_non_faces.append(np_img_cut)

    # Calcule the time
    end = time.time()
    print "\tTime spend to organize: ", end - start, 'seconds'

    return l_np_non_faces


def load_94_and_generate_faces(num_folders, size_image):

    l_class_figure = []

    l_np_faces = []

    s_path = "./data_sets/faces94/"

    l_up_folders = os.listdir(s_path)

    l_up_folders = l_up_folders[1:]

    for l_down_folders in l_up_folders:

        l_image_folders = os.listdir(s_path + l_down_folders)

        l_image_folders = l_image_folders[1:]

        if num_folders != 0:
            if num_folders <= len(l_image_folders):
                l_image_folders = l_image_folders[1:num_folders + 1]
            else:
                l_image_folders = l_image_folders[1:len(l_image_folders)]
        else:
            l_image_folders = l_image_folders[1:]

        for folder in l_image_folders:
            l_images_path = os.listdir(s_path + l_down_folders + "/" + folder + "/")

            i = 0
            for name_image in l_images_path:

                if ".gif" in name_image:
                    break

                if name_image[0] != '.':
                    # Instancie a new Class
                    l_class_figure.append(Figura_Class())

                    # Read the s_path of np_image
                    l_class_figure[-1].set_path(s_path + l_down_folders + "/" + folder + "/" + name_image)

                    # Load the np_image
                    np_image = load_image(s_path + l_down_folders + "/" + folder + "/" + name_image)

                    np_image = np_image[15:185, 20:160]

                    np_image = resize_img(np_image, size_image)

                    # show_img(np_image, size_image)

                    save_image(np_image, "./train/face/" + folder + "-" + str(i) + "-" + str(randrange(1, stop=10000)) + ".jpg", size_image, size_image)

                    l_class_figure[-1].set_image(np_image)

                    l_np_faces.append(np.array(np_image, dtype=np.uint8).reshape(-1))

                    i += 1

                    if i == 20:
                        break

    return l_class_figure, l_np_faces


def show_img(np_img, x=None, y=None):
    """
    Function that show the np_image without save it.
    :param img: np_image that will backend showed
    :param size_image: size of np_image
    :return: do not have a return
    """

    if y == None:
        y = x


    # Verify if it is a vector or a matrix
    if len(np_img.shape) == 1:
        # If it is a vector, reshape to a matrix
        img_reshaped = np_img.reshape((x, y))
    else:
        img_reshaped = np_img

    print img_reshaped.shape

    # Convert the numpy to a Image
    img_out = Image.fromarray(np.asarray(img_reshaped, dtype=np.uint8), "L")

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
    #x_range = range(x[0], x[1])
    #y_range = range(y[0], y[1])

    # Cut the np_image
    # matrix_cropped = img[y_range][:, x_range][:]
    matrix_cropped = img[y[0]: y[1], x[0]: x[1]]

    # Transform to a Image Object
    img_thumbnail = Image.fromarray(np.asarray(matrix_cropped, dtype=np.uint8), "L")

    # Resize the np_image
    img_thumbnail = img_thumbnail.resize((size_image, size_image), Image.ANTIALIAS)

    # Return the new np_image
    return np.array(img_thumbnail, dtype=np.uint8).reshape(-1)


def resize_img(img, size_image):
    """
    Receive a np_image from parameter and resize it to the CONST_size_image_algorithm
    :param img: Image
    :param size_image: new format
    :return:
    """
    #img_255 = np.asarray(img * 255, dtype=np.uint8)

    new_img = Image.fromarray(img, "L")

    # Resize it
    crop_img = new_img.resize((size_image, size_image), Image.ANTIALIAS)

    np_image = np.array(crop_img, dtype=np.uint8).reshape(-1)

    # Return the new np_image
    return np_image


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


def verify_args():
    """
    Procedure to verify the arguments from terminal
    :return:
    """
    if not len(sys.argv) == 11:
        print "[ERRO]: Invalid Parameters."
        print "\n[INFO]: Use: \nname_of_program.py  <batch_size>  <num_epochs>  <size_image_algorithm>  <learning_rate momentum>  " \
              "<train_again>  <num_files_94>  <directories_fddb_train>  <directories_fddb_test>  <directories_fddb_valid>"

        print "\n[DEFI]: batch_size: +integer."
        print "[DEFI]: num_epochs: +integer."
        print "[DEFI]: size_image_algorithm: +integer."
        print "[DEFI]: learning_rate: +float."
        print "[DEFI]: momentum: +float."
        print "[DEFI]: train_again: [y|n]."
        print "[DEFI]: num_files_94: 0 to all directories; integer for specific number."
        print "[DEFI]: directories_fddb_train: path."
        print "[DEFI]: directories_fddb_test: path."
        print "[DEFI]: directories_fddb_valid: path.\n\n"
        sys.exit()


def verify_acerts(l_faces_positions_original, region, found_face):
    x_region_actual = int(region[0])
    y_region_actual = int(region[1])

    is_x_inside_face = False
    is_y_inside_face = False

    index = 0

    min_x_face_original = 0
    min_y_face_original = 0

    for original_faces in l_faces_positions_original:
        center_x_face_original = float(original_faces[3])
        center_y_face_original = float(original_faces[4])
        radius = float(original_faces[0])

        # Is inside

        if abs(x_region_actual - center_x_face_original) < radius * 1.3:
            is_x_inside_face = True

        if abs(y_region_actual - center_y_face_original) < radius * 1.3:
            is_y_inside_face = True

        #if is_x_inside_face and is_y_inside_face:
        #    break

        if abs(x_region_actual - center_x_face_original) < abs(x_region_actual - float(
                l_faces_positions_original[min_x_face_original][3])):
            min_x_face_original = index

        if abs(y_region_actual - center_y_face_original) < abs(y_region_actual - float(
                l_faces_positions_original[min_y_face_original][4])):
            min_y_face_original = index

        index += 1

    if abs(x_region_actual - float(l_faces_positions_original[min_x_face_original][3])) < abs(
                    y_region_actual - float(l_faces_positions_original[min_y_face_original][4])):

        # difference = abs(x_region_actual - float(l_faces_positions_original[min_x_face_original][3]))
        distance = math.sqrt((x_region_actual - float(l_faces_positions_original[min_x_face_original][3])) ** 2 + (
            y_region_actual - float(l_faces_positions_original[min_x_face_original][4])) ** 2)

    else:
        #difference = abs(y_region_actual - float(l_faces_positions_original[min_y_face_original][4]))
        distance = math.sqrt((x_region_actual - float(l_faces_positions_original[min_y_face_original][3])) ** 2 + (
            y_region_actual - float(l_faces_positions_original[min_y_face_original][4])) ** 2)


    if found_face:
        if is_x_inside_face and is_y_inside_face:      # true_positive
            return 0, distance
        else:                          # false_positive
            return 1, distance
    else:
        if not (is_x_inside_face and is_y_inside_face): # true_negative
            return 2, distance
        else:                          # false_negative
            return 3, distance


def mark_result(np_img, position):
    """
    Procedure that mark the result of process
    :param image: np_image to mark
    :param position: position to mark
    :param s_path: s_path to save
    :return: do not have return
    """

    np_img_marked = np_img

    # np_img_marked = np.array(image)
    coords = position

    edge_x_axis = coords[1]
    edge_y_axis = coords[0]
    bound = coords[2]

    np_img_marked.flags.writeable = True

    # mark the y axis
    value = 255
    for y in range(edge_y_axis - bound, edge_y_axis):
        np_img_marked[edge_x_axis, y] = value
        np_img_marked[edge_x_axis - bound, y] = value
        np_img_marked[edge_x_axis, y + 1] = value
        np_img_marked[edge_x_axis - bound, y + 1] = value

    # Mark the x axis
    for x in range(edge_x_axis - bound, edge_x_axis):
        np_img_marked[x, edge_y_axis] = value
        np_img_marked[x, edge_y_axis - bound] = value
        np_img_marked[x + 1, edge_y_axis] = value
        np_img_marked[x + 1, edge_y_axis - bound] = value

    return np_img_marked


def make_graphic(l_batch, num_batch):

    labels = ["True Positives", "False Positives", "True Negatives", "False Negatives"]

    plt.figure(dpi=300)

    plt.ylabel("Distance of the Central Point")
    plt.xlabel("Type of Erros")
    plt.title("Distance Graph " + str(num_batch))

    #plt.xlim(0, 6)
    #plt.ylim(-5, 80)
    plt.legend(loc="upper left")

    # For each batch
    plt.boxplot(l_batch, labels=labels, notch=True)

    plt.savefig("boxplot-" + str(num_batch) + ".png")


def analyze_results(l_out, l_Figura_class, num_batch):
    print "[INFO]: Analyzing the Results"


    l_statistics_batch = []

    l_statistics_batch.append([])
    l_statistics_batch.append([])
    l_statistics_batch.append([])
    l_statistics_batch.append([])

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
            sys.stdout.write("\r\tProcessed: " + str(num_region + 1) + ":" + str(index + 1) + " of " + str(len(l_out)) +
                             ". \tCompleted: " + str(index + 1 / float(index + 1) * 100.0) +
                             "%.\tTime spend until now: " + str(time.time() - start) + "s")
            sys.stdout.flush()

            if region[0] > 0.5:
                image_marked = True

                np_img = l_Figura_class[index].get_image()

                np_img_cut = mark_result(np_img, l_Figura_class[index].l_faces_positions_regions[num_region])

                l_Figura_class[index].set_image(np_img_cut)
                error_type, distance = verify_acerts(l_Figura_class[index].l_faces_positions_original,
                                                        l_Figura_class[index].l_faces_positions_regions[num_region], 1)
            else:
                error_type, distance = verify_acerts(l_Figura_class[index].l_faces_positions_original,
                                                        l_Figura_class[index].l_faces_positions_regions[num_region], 0)

            if error_type == 0:
                l_statistics_batch[0].append(distance)
            elif error_type == 1:
                l_statistics_batch[1].append(distance)
            elif error_type == 2:
                l_statistics_batch[2].append(distance)
            elif error_type == 3:
                l_statistics_batch[3].append(distance)

            if image_marked:
                save_image(l_Figura_class[index].get_image(),
                                        "./out/f/f" + str(index) + "-" + str(num_region) + ".jpg")
                image_marked = False

            num_region += 1

            if num_region >= len(l_Figura_class[index].l_faces_positions_regions):
                break

        index += 1

    make_graphic(l_statistics_batch, num_batch)

    print "\n\tTrue Positives: ", len(l_statistics_batch[0]), "\tFalse Positives: ", \
        len(l_statistics_batch[1]), "\n\tTrue Negatives: ", \
        len(l_statistics_batch[2]), "\tFalse Negatives: ", len(l_statistics_batch[3])

    save_results(l_out)

    end = time.time()


    print "\n\tTime spend to generate the outputs: ", end - start, 'seconds', "\n"


def read_paths(path):

    paths = []

    # Open the file of directories
    file = open(path, "r")

    for line in file:
        # Create a list of diretories
        paths.append(line)

    return paths
