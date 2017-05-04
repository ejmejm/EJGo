import tensorflow as tf
import numpy as np

board_size = 19

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 5

x = tf.placeholder(tf.float32, [None, board_size * board_size])
y = tf.placeholder(tf.float32, [None, 2])

def nn_forward(data):
    print("TEST1111111111")
    hidden_1_layer = {"weights": tf.Variable(tf.random_normal([board_size * board_size, n_nodes_hl1])),
    "biases": tf.Variable(tf.random_normal([n_nodes_hl1]))}

    print("TEST2")
    hidden_2_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    "biases": tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    "biases": tf.Variable(tf.random_normal([n_nodes_hl3]))}

    out_layer = {"weights": tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    "biases": tf.Variable(tf.random_normal([n_classes]))}

    print("TEST3")
    l1 = tf.add(tf.matmul(data, hidden_1_layer["weights"]), hidden_1_layer["biases"])
    print("TEST4ghchgfhdfghfghfghfghfghh")
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(data, hidden_2_layer["weights"]), hidden_2_layer["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(data, hidden_3_layer["weights"]), hidden_3_layer["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(data, output_layer["weights"]) + output_layer["biases"]

    return output

def train_neural_network(x, gameData):
    prediction = nn_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for index in range(int(len(gameData/batch_size))):
                board = np.zeroes(board_size * board_size)
                for node in gameData[batch_size * epoch + index].get_main_sequence():
                    board = -board; # Changes player perspective, black becomes white and vice versa
                    if node.get_move()[1] != None:
                        next_move = []
                        next_move.append(node.get_move()[1])
                        next_move = np.array(next_move) # y = an array in the form [board_x_position, board_y_position]
                        _, c = sess.run([optimizer, cost], feed_dict = {x: board, y: next_move})
                        epoch_loss += c
                        board[y[0] * board_size + y[1]] = 1 # Update board with new move
                    # TODO: Train on passes here?
                    print("Epoch ", epoch, " completed out of ", hm_epochs, ", Loss: ", epoch_loss)

        print("Prediction: ", prediction)
        print("Acutal: ", y)

def test_network(gameData):
    train_neural_network(x, gameData)
