##### regenerate images for class 'o' or N class to create balanced data ###################
############## ALREADY DONE DO NEED TO RUN ONCE AGAIN
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import os
import config
# load the input image, convert it to a NumPy array, and then
# reshape it to have an extra dimension
print("[INFO] loading example image...")

train_data_N = config.TRAIN_DATA_PATH # '/floyd/home/datasets/orig/DATASET/TRAIN/N'
new_data_N = config.TRAIN_DATA_PATH # '/floyd/home/datasets/orig/DATASET/TRAIN/N'
#image = load_img('/floyd/home/datasets/test/00000281.jpg')
paths = os.listdir(train_data_N)
for imagepath in paths:
    image = load_img(train_data_N +"/"+imagepath)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    # construct the image generator for data augmentation then
    # initialize the total number of images generated thus far
    aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,height_shift_range=0.1, 
                shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")
    total = 0

    # construct the actual Python generator
    #print("[INFO] generating images...")
    imageGen = aug.flow(image, batch_size=1, save_to_dir=new_data_N,save_prefix='new_aug',
                        save_format="jpg")

    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1

        # if we have reached 10 examples, break from the loop
        if total == 4:
            break