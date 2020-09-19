# import the necessary package
from keras.callbacks import Callback
import os
import datetime

class EpochCheckpoint(Callback):
	def __init__(self, outputPath, every=5, startAt=0):
		# call the parent constructor
		super(Callback, self).__init__()

		# store the base output path for the model, the number of
		# epochs that must pass before the model is serialized to
		# disk and the current epoch value
		self.outputPath = outputPath
		self.every = every
		self.intEpoch = startAt

	def on_epoch_end(self, epoch, logs={}):
		# check to see if the model should be serialized to disk
		if (self.intEpoch > 0) and ((self.intEpoch ) % self.every == 0 ):
			p = os.path.sep.join([self.outputPath,
				"epoch_{}.hdf5".format(str(self.intEpoch) +"_"+ str(datetime.date.today()))])
			self.model.save(p, overwrite=True)

		# increment the internal epoch counter
		self.intEpoch += 1