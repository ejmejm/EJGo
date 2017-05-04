import tensorflow as tf
import numpy as np
import time

board_size = 19

n_nodes_hl1 = 250
n_nodes_hl2 = 250
n_nodes_hl3 = 250

n_classes = board_size * board_size
batch_size = 5

x = tf.placeholder(tf.float32, [None, board_size * board_size])
y = tf.placeholder(tf.float32, [None, board_size * board_size])

def nn_forward(data):
    hidden_1_layer = {"weights": tf.Variable(tf.random_normal([board_size * board_size, n_nodes_hl1])),
    "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}
    #print(tf.shape(hidden_1_layer["weights"]))

    hidden_2_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    "biases": tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    l1 = tf.nn.tanh(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.tanh(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.tanh(l3)

    output = tf.matmul(l3, output_layer["weights"]) + output_layer["biases"]

    return output

def load(save_path):
    prediction = nn_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path=save_path)
    return sess


def train_neural_network(x, gameData):
    prediction = nn_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    save_path = "checkpoints/next_move_model"

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for index in range(len(gameData)):
                board = np.zeros(board_size * board_size).reshape(1, board_size * board_size)
                for node in gameData[index].get_main_sequence():
                    board = -board; # Changes player perspective, black becomes white and vice versa
                    if node.get_move()[1] != None:
                        next_move = np.zeros(board_size * board_size).reshape(1, board_size * board_size)
                        next_move[0][node.get_move()[1][0] * board_size +
                        node.get_move()[1][1]] = 1 # y = an array in the form [board_x_position, board_y_position]
                        _, c = sess.run([optimizer, cost], feed_dict = {x: board, y: next_move})
                        epoch_loss += c
                        board[0][node.get_move()[1][0] * board_size + node.get_move()[1][1]] = 1 # Update board with new move
                    # TODO: Train on passes here?
            print("Epoch ", epoch+1, " completed out of ", hm_epochs, ", Loss: ", epoch_loss)
        saver.save(sess=sess, save_path=save_path)

def train(gameData):
    train_neural_network(x, gameData)

def get_move(board):
    prediction = nn_forward(x)
    sess = load("checkpoints/next_move_model")
    return sess.run(prediction, feed_dict = {x: board})
