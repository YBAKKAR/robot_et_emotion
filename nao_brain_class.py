from __future__ import division, absolute_import

import sys
from os.path import isfile

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


class EDCNN:

    def __init__(self):
        self.target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    def build_network(self):
        """
        Build the convnet.
        Input is 48x48
        3072 nodes in fully connected layer
        """
        #print("\n---> Starting Neural Network \n")
        self.network = input_data(shape=[None, 48, 48, 1])
        #print("Input data", self.network.shape[1:])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        #print("Conv1", self.network.shape[1:])
        self.network = max_pool_2d(self.network, 3, strides=2)
        #print("Maxpool", self.network.shape[1:])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        #print("Conv2", self.network.shape[1:])
        self.network = max_pool_2d(self.network, 3, strides=2)
        #print("Maxpool2", self.network.shape[1:])
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        #print("Conv3", self.network.shape[1:])
        self.network = dropout(self.network, 0.3)
        #print("Dropout", self.network.shape[1:])
        self.network = fully_connected(self.network, 3072, activation='relu')
        #print("Fully connected", self.network.shape[1:])
        self.network = fully_connected(self.network, len(self.target_classes), activation='softmax')
        #print("Output", self.network.shape[1:])
        #print('\n')
        self.network = regression(self.network, optimizer='momentum', metric='accuracy',
                                  loss='categorical_crossentropy')
        self.model = tflearn.DNN(self.network, checkpoint_path='model_1_kaggle', max_checkpoints=1,
                                 tensorboard_verbose=3)
        self.load_model()

    def predict(self, image):
        """
        Image is resized to 48x48
        model.predict() is an inbuilt function in tflearn.
        """
        if image is None:
            return None
        image = image.reshape([-1, 48, 48, 1])
        return self.model.predict(image)

    def load_model(self):
        """
        Loads pre-trained model.
        model.load() is an inbuilt function in tflearn
        """
        if isfile("model_1_kaggle.tflearn.meta"):
            self.model.load("model_1_kaggle.tflearn")
            print('\n---> CNN charge en memoire!')
        else:
            print("---> Pas de fichier pour loader CNN")


if __name__ == "__main__":
    print("\n------------ROBOT et EMOTION------------\n")
    import traitement
