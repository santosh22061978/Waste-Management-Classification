# Waste-Management-Classification
Intelligent Waste Management Classification System using Computer Vision with Deep Learning


Some facts about waste management-
Waste management is the precise name for the collection, transportation, disposal or recycling and monitoring of waste. This term is assigned to the material, waste material that is produced through human being activity. This material is managed to avoid its adverse effect over human health and environment. The world bank report showed that there are almost 4 billion tons of waste around the world every year and the urban alone contributes a lot to this number, the waste is predicted to increase by 70 percent in the year 2025.
The most important reason for proper waste management is to protect the environment and for the health and safety of the population. Reduce the volume of the solid waste stream through the implementation of waste reduction and recycling programs. waste can cause air and water pollution. Most of the wastes end up in landfills. This leads to many issues like
•	Increase in landfills
•	Eutrophication
•	Consumption of toxic waste by animals
•	Leachate
•	Increase in toxins
•	Land, water and air pollution
It is important to have an advanced/intelligent waste management system to manage a variety of waste materials. One of the most important steps of waste management is the separation of the waste into the different components and this process is normally done manually by hand-picking. To simplify the process, we propose an intelligent  waste  material  classification  system, which can classify into three classes (Organic , recyclable and Non-Recyclable ). Further we can automate this process using IOT and machine learning.
For this work we are using public dataset which is available on Kaggle website (https://www.kaggle.com/sapal6/waste-classification-data-v2). This is a medium dataset and consist of 27982 images , which is divided into three different classes Organic , recyclable and Non-Recyclable, all the pictures of the images need to resize down to fixed size something like 126*126. Few samples of the images are shown below. 
     

For the pre-processing stage, data augmentation method was performed on the images, because of the unbalanced dataset. This technique was chosen because of the different orientations of the waste materials. Some of the technique includes, random of the image, translating the image, randomly scaling the image, image shearing, randomly scaling of the image. With this technique it maximizes the dataset size and also help to develop the model more generalized. Most of the images have lots of extra white background which also need to remove as a part of preprocessing. Further to get better result we can apply aspect aware preprocessing, mean preprocessing, cropping etc. for each image available in dataset. To create balance dataset, we may need to apply under sampling technique or class weight mechanism to achieve better result during model development.
First divide the data in training (80%) and testing (20%). Further takes 10% of data from training ( after the split of training ) for the use of validation.
 

CNNs are typically trained “end-to-end” meaning that the entire network is trained to optimize the problem of interest, i.e., all the parameters of the network are adjusted until the peak classification rate is achieved. As here we do not have large dataset, so we will use transfer learning either feature extraction or Fine tuning. First, we are using the CNN as an intermediary feature extractor from pre-trained network architecture ( Resnet 50 or VGG 16)  using Lower level . The downstream machine learning classifier ( like linear SVM ) will take care of learning the underlying patterns of the features extracted from the CNN. 
 



Next, we will do the Fine tuning to compare the accuracy of the model.
Fine-tuning is a type of transfer learning. We apply fine-tuning( i.e. network surgery ) to deep learning models that have already been trained on a given dataset. Typically, these networks are state-of-the-art architectures such as VGG, ResNet, and Inception that have been trained on the ImageNet dataset. Each of the images must be transformed into a fixed size image to match the required input size of the network structure.

To optimize I/O operation and proper memory utilization during model training we will convert the dataset into HDF5 format.  






We train the model on input dataset ( HDF5 ) using transfer learning and classify the data. Model development will be done using python, Keras(deep learning library using tensorflow). 
Another Approach with supervise learning
Incase transfer learning does not able to solve our problem then we have to design the network from scratch.
However, traditional feature extraction methods can only extract some low-level features of images, and prior knowledge is necessary to select useful features, which can be greatly affected by humans. Deep learning techniques can extract high-level abstract features from images automatically using convolutional neural network.

Exactly which method performs best is entirely dependent on dataset and classification problem. 
The Model output will predict probability score and class label for the given input.
Finally, we will compare all three models, and which ever give better accuracy will choose the final one.
