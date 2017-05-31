import tensorflow as tf
import numpy as np
import time
import board as go_board

board_size = 19

n_classes = board_size * board_size

sess = tf.Session()

x = tf.placeholder(tf.float32, [None, board_size * board_size])
y = tf.placeholder(tf.float32, [None, board_size * board_size])

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def cnn_forward(data):
    weights = {"W_conv1": tf.Variable(tf.random_normal([19, 19, 1, 64])),
    "W_conv2": tf.Variable(tf.random_normal([19, 19, 128, 128])),
    "W_fc": tf.Variable(tf.random_normal([19*19*128, 1024])),
    "out": tf.Variable(tf.random_normal([1024, n_classes]))}

    weights = {"b_conv1": tf.Variable(tf.random_normal([64])),
    "b_conv2": tf.Variable(tf.random_normal([128])),
    "b_fc": tf.Variable(tf.random_normal([1024])),
    "out": tf.Variable(tf.random_normal([1024, n_classes]))}

    data = tf.reshape(data, shape=[-1, 19, 19, 1])

    conv1 = conv2d(data, weights["W_conv1"])
    conv1 = maxpool2d(conv1)

    conv2 = conv2d(conv1, weights["W_conv2"])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 19*19*128])
    fc = tf.nn.tanh(tf.matmul(fc, weights["W_fc"]) + biases["b_fc"])

    output = tf.matmul(fc, weights["out"]) + biases["out"]

    return output

def load(save_path):
    pred = nn_forward(x)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path=save_path)
    prediction = pred
    return {"session": sess, "prediction": pred}


def train_neural_network(x, gameData):
    prediction = nn_forward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    save_path = "checkpoints/next_move_model.ckpt"

    hm_epochs = 5
    batch_size = len(gameData)/5

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            game_loss = 0
            game_index = 0
            game_counter = 0
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
                        game_loss += c
                        board[0][node.get_move()[1][0] * board_size + node.get_move()[1][1]] = 1 # Update board with new move
                game_counter += 1
                if game_counter % batch_size == 0:
                    game_index += 1
                    print("Epoch", epoch+1, ", Game batch", game_index, "completed, Loss:", game_loss)
                    game_loss = 0
                    game_counter = 0

            saver.save(sess=sess, save_path=("checkpoints/nm_epoch_" + str(epoch+1) + ".ckpt"))
            print("\nEpoch", epoch+1, "completed out of", hm_epochs, ", Loss:", epoch_loss, "\n")
        saver.save(sess=sess, save_path=save_path)

def train(gameData):
    train_neural_network(x, gameData)

def get_prob_board(board, model):
    board = board.reshape(1, board_size * board_size)

    move = sess.run(model["prediction"], feed_dict = {x: board})
    return move

def predict_move(board, model, level=0, bot_tile=1):
    c_board = np.copy(board.reshape(board_size * board_size))
    prob_board = get_prob_board(c_board, model).reshape(board_size * board_size)
    sorted_board = np.asarray(sorted(enumerate(prob_board), reverse = True, key=lambda i:i[1]))

    move_found = False
    i = 0
    while move_found == False:
        if i >= len(c_board):
            move_found = True
            return(-1)
        if c_board[int(sorted_board[i][0])] == 0 and go_board.make_move(c_board.reshape(board_size, board_size), np.array([int(sorted_board[i][0]/board_size), int(sorted_board[i][0] % board_size)]), bot_tile) != None:
            c_board[int(sorted_board[i][0])] = 1
            move_found = True
            return np.array([int(sorted_board[i][0]/board_size), int(sorted_board[i][0] % board_size)])
        i += 1