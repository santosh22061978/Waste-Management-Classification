# USAGE
# python build_dataset.py

# import the necessary packages
import config
from imutils import paths
import random
import shutil
import os
from aspectawarepreprocessor import AspectAwarePreprocessor
#from imagetoarraypreprocessor import ImageToArrayPreprocessor
import cv2

aap = AspectAwarePreprocessor(128, 128)
# grab the paths to all input images in the original input directory
# and shuffle them
imagePaths = sorted(list(paths.list_images('/floyd/home/datasets/orig/DATASET')))
random.seed(42)
random.shuffle(imagePaths)

# compute the training and testing split
i = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:i]
testPaths = imagePaths[i:]

# we'll be using part of the training data for validation
i = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:i]
trainPaths = trainPaths[i:]

# define the datasets that we'll be building
datasets = [
	("training", trainPaths, config.TRAIN_PATH),
	("validation", valPaths, config.VAL_PATH),
	("testing", testPaths, config.TEST_PATH)
]

# loop over the datasets
for (dType, imagePaths, baseOutput) in datasets:
	# show which data split we are creating
	print("[INFO] building '{}' split".format(dType))

	# if the output base output directory does not exist, create it
	if not os.path.exists(baseOutput):
		print("[INFO] 'creating {}' directory".format(baseOutput))
		os.makedirs(baseOutput)

	# loop over the input image paths
	for inputPath in imagePaths:
		# extract the filename of the input image and extract the
		# class label ("0" for "negative" and "1" for "positive")
		filename = inputPath.split(os.path.sep)[-1]
		#print(filename) 
		label = inputPath.split(os.path.sep)[-2]
		if(label == 'N' ):
			label ='0'
		elif( label == 'O' ):
			label ='1'
		else:
			label ='2'
		#print(label)
        

		# build the path to the label directory
		labelPath = os.path.sep.join([baseOutput, label])

		# if the label output directory does not exist, create it
		if not os.path.exists(labelPath):
			print("[INFO] 'creating {}' directory".format(labelPath))
			os.makedirs(labelPath)
			
		file_extension = os.path.splitext(inputPath)[1]
		#print(str(file_extension).lower)
		if((str(file_extension)).lower() =='.jpg'):
			image = cv2.imread(inputPath)
			image = aap.preprocess(image)
			# construct the path to the destination image and then copy
			# the image itself
			p = os.path.sep.join([labelPath, filename])
			cv2.imwrite(p, image) 
			#shutil.copy2(inputPath, p)