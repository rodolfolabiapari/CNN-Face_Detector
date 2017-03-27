
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


NUM_IMAGES_TRAINED = -1
#CONST_batch_size = 4096          # The CONST_batch_size cannot backend bigger
# than amount images TODO DEVE SER AUTO
# CONST_batch_size = 3084
# CONST_batch_size = 2048
#CONST_batch_size = 1024
CONST_batch_size = 512
#CONST_batch_size = 64
CONST_num_epochs = 1250  # Passages on through the dataset
CONST_size_image_algorithm = 36  # Size of training np_image
CONST_learning_rate = 0.01
CONST_momentum = 0.9
# CONST_train_again = basic_functs.verify_args()       # TODO arrumar o args
CONST_train_again = "y"
# TODO
num_files_94 = 0
