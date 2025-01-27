# DEEP-LEARNING-PROJECT
COMPANY: CODTECH IT SOLUTIONS

NAME: RITHI RYTHAMI S

INTERN ID: CT08KFD

DOMAIN: DATA SCIENCE

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION:This Python script is an end-to-end deep learning algorithm for image classification using the CIFAR-10 dataset, a general-purpose dataset containing 60,000 32x32 color images from 10 different classes, including airplanes, cars, birds, cats, dogs, and more. This application leverages the PyTorch library to define, train, measure, and deploy a convolutional neural network (CNN) model that can classify these images into their respective categories.


Use transformations to normalize the image and convert it to a tensor via PyTorch. The CIFAR-10 dataset is then split into training and testing and loaded using the DataLoader class for batch processing. The training data is shuffled to improve the generalization model, while the testing data remains unchanged for evaluation. The network consists of two convolutional layers to extract  features from the image, each of which has ReLU activation and max pooling to reduce the dimensionality and capture important patterns. After the feature map is flattened, we process the features through all layers and finally generate an output vector of size 10 corresponding to  10 batches of images. 



The network uses the ReLU activation function to ensure non-linearity and achieve better learning. For each batch, the input image is passed through the network and the predicted value is compared with the real text using the cross-entropy loss function. The loss is compensated by the network and the optimizer adjusts the weight to reduce the error. The scenario also tracks learning losses at every moment to track learning progress. Measure the performance of the model on the test data by calculating model accuracy. The training examples are then sent to individual custom images.


These images are preprocessed using the same transformations as the training data, passed through the network, and given predicted class labels. Send. Demonstrates the power of  neural networks in image classification and demonstrates that trained models can be integrated into real-world scenarios to make predictions about unseen objects.
