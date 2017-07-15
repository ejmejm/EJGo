import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

def get_network():
    network = input_data(shape=[None, 19, 19, 2], name='input')
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")
    network = local_response_normalization(network)
    network = conv_2d(network, 128, 3, activation='relu', regularizer="L2")
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 3, activation='relu', regularizer="L2")
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 3, activation='relu', regularizer="L2")
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 3, activation='relu', regularizer="L2")
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.6)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.6)
    network = fully_connected(network, 19*19, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    return network
