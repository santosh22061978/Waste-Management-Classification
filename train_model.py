# USAGE
# python train_model.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from wastenet import WasteNet
import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
from keras.callbacks import ModelCheckpoint
import os



# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 70
INIT_LR = 1e-3 # 1e-2
BS = 32

# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images(config.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals
print("printing class weight to account for skew in the labeled data")
print(classWeight)
# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=15,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.05,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(128, 128),
	color_mode="rgb",
	shuffle=True,
	batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(128, 128),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(128, 128),
	color_mode="rgb",
	shuffle=False,
	batch_size=BS)

path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])

checkpoint = ModelCheckpoint(config.Model_PATH, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks = [checkpoint, TrainingMonitor(path)]
#callbacks_list = [checkpoint]
# initialize our CancerNet model and compile it
model = WasteNet.build(width=128, height=128, depth=3,	classes=3)
#opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.6))
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# fit the model
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // BS,
	validation_data=valGen,
	validation_steps=totalVal // BS,
	#class_weight=classWeight, without class weight adam optimiser
    callbacks=callbacks,
	epochs=NUM_EPOCHS,verbose=1)

print("saving the model")
model.save('/floyd/home/output/Vgg_waste_manage_ADAM_noWeight_70.model', overwrite=True)
# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,	steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,target_names=testGen.class_indices.keys()))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("/floyd/home/output/lossplot_adam_noweight_70")