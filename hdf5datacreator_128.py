from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from aspectawarepreprocessor import AspectAwarePreprocessor
from hdf5datasetwriter import HDF5DatasetWriter
import config
import pandas as pd
import cv2
import numpy as np
#from sklearn.model_selection import train_test_split
#from AspectAwarePreprocessor import AspectAwarePreprocessor
#from HDF5DatasetWriter import HDF5DatasetWriter
#from imutils import paths
import json

df = pd.read_csv( config.BASE_PATH +'/train.csv')

#shuffle dataframe in-place and reset the index
df = df.sample(frac=1).reset_index(drop=True)
# grab the paths to the images
trainPaths = df.path.values.tolist()
trainLabels = df.category.values.tolist()
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perform stratified sampling from the training set to build the
# testing split from the training data
#split = train_test_split(trainPaths, trainLabels,test_size=NUM_VAL_IMAGES, stratify=trainLabels,random_state=42)
#(trainPaths, testPaths, trainLabels, testLabels) = split

# perform another stratified sampling, this time to build the
# validation data
split = train_test_split(trainPaths, trainLabels,test_size=5868, stratify=trainLabels,random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

# grab the test paths to the images
dftest = pd.read_csv(config.BASE_PATH +'/test.csv')

#shuffle dataframe in-place and reset the index
dftest = dftest.sample(frac=1).reset_index(drop=True)

testPaths = dftest.path.values.tolist()
testLabels = dftest.category.values.tolist()
le = LabelEncoder()
testLabels = le.fit_transform(testLabels)

# define the path to the output training, validation, and testing
# HDF5 files
TRAIN_HDF5 = config.TRAIN_HDF5 
VAL_HDF5 = config.VAL_HDF5
TEST_HDF5 = config.TEST_HDF5 
# construct a list pairing the training, validation, and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
    ("train", trainPaths, trainLabels, TRAIN_HDF5),
    ("val", valPaths, valLabels, VAL_HDF5),
    ("test", testPaths, testLabels, TEST_HDF5)]

# initialize the image pre-processor and the lists of RGB channel
# averages
import datetime
print(datetime.datetime.now())
aap = AspectAwarePreprocessor(128, 128)
(R, G, B) = ([], [], [])


# loop over the dataset tuples
for (dType, paths, labels, outputPath) in datasets:
    # create HDF5 writer
    print("[INFO] building {}...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 128, 128, 3), outputPath)

    # initialize the progress bar
    #widgets = ["Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    #pbar = progressbar.ProgressBar(maxval=len(paths),widgets=widgets).start()

    # loop over the image paths
    for (i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # if we are building the training dataset, then compute the
        # mean of each channel in the image, then update the
        # respective lists
        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add the image and label # to the HDF5 dataset
        writer.add([image], [label])
        #pbar.update(i)

    # close the HDF5 writer
    #pbar.finish()
    writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing means...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
print(datetime.datetime.now())