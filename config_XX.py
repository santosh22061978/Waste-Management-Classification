# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "/floyd/home/datasets/input/dataset"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and validation split
BASE_PATH = "/floyd/home/datasets" 

# derive the training, validation, and testing directories
TRAIN_PATH = "/floyd/home/datasets/training"
VAL_PATH = "/floyd/home/datasets/validation"
TEST_PATH = os.path.sep.join([BASE_PATH, "test"])

# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.15

#output model save
Model_PATH ="/floyd/home/output/Vgg_waste_manage_tune.model" 
# since we do not have validation data or access to the testing
# labels we need to take a number of images from the training
# data and use them instead
NUM_CLASSES = 3
NUM_VAL_IMAGES = 5868 
NUM_TRAIN_IMAGES = 23475  

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = "/floyd/home/datasets/train.hdf5"
VAL_HDF5 = "/floyd/home/datasets/val.hdf5"
TEST_HDF5 = "/floyd/home/datasets/test.hdf5"

# define the path to the dataset mean
DATASET_MEAN = "/floyd/home/output/waste_mean.json"
OUTPUT_PATH = "/floyd/home/output"
MODEL_CHECKPOINT_PATH = "/floyd/home/output"