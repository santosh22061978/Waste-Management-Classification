# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
#from aspectawarepreprocessor import AspectAwarePreprocessor
#from simpledatasetloader import SimpleDatasetLoader
from simplepreprocessor import SimplePreprocessor
from patchpreprocessor import PatchPreprocessor
from meanpreprocessor import MeanPreprocessor
from hdf5datasetgenerator import HDF5DatasetGenerator
from fcheadnet import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from trainingmonitor import TrainingMonitor
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
#from imutils import paths
import numpy as np
import json
import os
import config
from epochcheckpoint import EpochCheckpoint


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
sp = SimplePreprocessor(128, 128)
#pp = PatchPreprocessor(224, 224)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()
print("initialize the preprocessors done")
classNames = {0: "Non-Recyclable", 1: "Organic", 2: "Recyclable"}
#classes=len(classNames)
# initialize the training and validation dataset generators
#trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug,preprocessors=[pp, mp, iap], classes=len(classNames))
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug,preprocessors=[sp,mp, iap], classes=3)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 32,preprocessors=[sp, mp, iap], classes=3)
#********************

print(" initialize the training and validation dataset generators")


# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet", include_top=False,input_tensor=Input(shape=(128, 128, 3)))

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

# # loop over all layers in the base model and freeze them so they
# # will *not* be updated during the training process
# for layer in baseModel.layers:
# 	layer.trainable = False

# # compile our model (this needs to be done after our setting our
# # layers to being non-trainable
# print("[INFO] compiling model...")
# opt = RMSprop(lr=0.001)
# model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])

# # train the head of the network for a few epochs (all other
# # layers are frozen) -- this will allow the new FC layers to
# # start to become initialized with actual "learned" values
# # versus pure random
# # construct the set of callbacks
# path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
# callbacks = [TrainingMonitor(path)]
# class_weight = {0:14,1: 1,2: 2}
# print("[INFO] training head...")
# train the network
# model.fit_generator(trainGen.generator(),steps_per_epoch=trainGen.numImages // 32,validation_data=valGen.generator(),
#                     validation_steps=valGen.numImages // 32,epochs=3,max_queue_size=10,
#                     callbacks=callbacks,verbose=1, class_weight=class_weight)


# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[13:]:
	layer.trainable = True

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling model...")
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])

callbacks = [EpochCheckpoint(config.MODEL_CHECKPOINT_PATH, every=2,startAt=5),TrainingMonitor(path)]
#class_weight = {0:10,1: 1,2: 2}
print("[INFO] training head...")
# train the network
#opt = SGD(lr=0.0008) # results were not satisfactory using new value
opt = SGD(lr=0.0006)
model.compile(loss="categorical_crossentropy", optimizer=opt,	metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
print("[INFO] fine-tuning model...")

model.fit_generator(trainGen.generator(),steps_per_epoch=trainGen.numImages // 32,validation_data=valGen.generator(),
                    validation_steps=valGen.numImages // 32,epochs=50,max_queue_size=10,
                    callbacks=callbacks,verbose=1)

model.save(config.Model_PATH, overwrite=True)

# close the HDF5 datasets
trainGen.close()
valGen.close()