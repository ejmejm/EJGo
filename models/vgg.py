# Source: https://github.com/tflearn/tflearn/blob/master/examples/images/vgg_network.py

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import global_vars_go as gvg

def get_network():
    network = input_data(shape=[None, 19, 19, 2], name="input")

    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')

    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')

    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2)

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, gvg.board_size * gvg.board_size, activation='softmax')

    network = regression(network, optimizer='rmsprop',
                     loss='categorical_crossentropy',
                     learning_rate=0.0001, name="target")

    return network
