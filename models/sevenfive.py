import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import global_vars_go as gvg

def get_network():
    network = input_data(shape=[None, gvg.board_size, gvg.board_size, gvg.board_channels], name='input')
    network = conv_2d(network, 64, 7, activation='relu')
    network = conv_2d(network, 64, 5, activation='relu')
    network = conv_2d(network, 64, 5, activation='relu')
    network = conv_2d(network, 48, 5, activation='relu')
    network = conv_2d(network, 48, 5, activation='relu')
    network = conv_2d(network, 32, 5, activation='relu')
    network = conv_2d(network, 32, 5, activation='relu')
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, gvg.board_size * gvg.board_size, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    return network
