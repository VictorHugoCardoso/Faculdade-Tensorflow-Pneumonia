from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
from PIL import Image
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
train_list = 'train_list.txt'
test_list = 'test_list.txt'
size=256

from tflearn.data_utils import image_preloader
X, Y = image_preloader(train_list, image_shape=(size, size), mode='file', grayscale=True,categorical_labels=True, normalize=True)
test_x, test_y = image_preloader(test_list, image_shape=(size, size), mode='file', grayscale=True,categorical_labels=True, normalize=True)

X = np.reshape(X, (-1, size, size, 1))
test_x = np.reshape(test_x, (-1, size, size, 1))

network = input_data(shape=[None, size, size, 1])
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='momentum',loss='categorical_crossentropy',learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0, tensorboard_dir='pneumonia5')
'''
model.fit(X, Y, validation_set=0.1, n_epoch=6, snapshot_step=100, show_metric=True, run_id='pneumonia')
model.save('modelo.tflearn')
'''
model.load('./modelo.tflearn')

print('================================')
print(model.predict([test_x[0],test_x[1]]))