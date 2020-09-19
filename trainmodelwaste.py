# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from wastecnn import WasteCNNNet
from simplepreprocessor import SimplePreprocessor
from patchpreprocessor import PatchPreprocessor
from meanpreprocessor import MeanPreprocessor
from hdf5datasetgenerator import HDF5DatasetGenerator
from fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from trainingmonitor import TrainingMonitor
#from keras.optimizers import RMSprop
#from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
#from imutils import paths
import numpy as np
import json
import os
import config
from epochcheckpoint import EpochCheckpoint
import datetime


# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())
print("initialize the preprocessors")
# initialize the image preprocessors
sp = SimplePreprocessor(224, 224)
#pp = PatchPreprocessor(128, 128)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()
print("initialize the preprocessors done")
classNames = {0: "Non-Recyclable", 1: "Organic", 2: "Recyclable"}
#classes=len(classNames)
# initialize the training and validation dataset generators


print(" initialize the training and validation dataset generators")

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug,preprocessors=[sp,mp, iap], classes=3)

valGen = HDF5DatasetGenerator(config.VAL_HDF5, 32,preprocessors=[sp, mp, iap], classes=3)
#********************


# initialize the optimizer
print("[INFO] compiling model...")
opt = Adam(lr=1e-3)
#model = WasteCNNNet.build(width=128, height=128, depth=3,	classes=3, reg=0.0002)

baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

for layer in baseModel.layers:
 	layer.trainable = False

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)



path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])

callbacks = [EpochCheckpoint(config.MODEL_CHECKPOINT_PATH, every=2,startAt=5),TrainingMonitor(path)]

model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
print(model.summary())
print("[INFO] Training model...")
print(trainGen.numImages)
print(valGen.numImages)
print(datetime.datetime.now())

model.fit_generator(trainGen.generator(),steps_per_epoch=trainGen.numImages // 32,validation_data=valGen.generator(),
                    validation_steps=valGen.numImages // 32,epochs=40,max_queue_size=10,
                    callbacks=callbacks,verbose=1)

print(datetime.datetime.now())
model.save(config.Model_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()