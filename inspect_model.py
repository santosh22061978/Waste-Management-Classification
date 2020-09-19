# USAGE
# python inspect_model.py --include-top -1

# import the necessary packages
from keras.applications import VGG16
from keras.activations import 

# load the VGG16 network
print("[INFO] loading network...")
model = VGG16(weights="imagenet",	include_top=True)
print("[INFO] showing layers...")

# loop over the layers in the network and display them to the
# console
for (i, layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i, layer))