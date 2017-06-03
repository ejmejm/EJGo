import tensorflow as tf
import numpy as np
import time
import board as go_board
import global_vars_go
import matplotlib.pyplot as plt
from sgfmill.sgfmill import sgf_moves

mode = "cnn"
board_size = 19

n_nodes_hl1 = 300
n_nodes_hl2 = 300
n_nodes_hl3 = 300

batch_size = 200 # How many board states (not full games) to send to GPU at once, this is about the max with my GPU's RAM
batch_display_stride = 200 # How many batches to send to GPU before displaying a visual update

n_classes = board_size * board_size

sess = tf.Session()

# IO placeholders
x = tf.placeholder(tf.float32, [None, board_size * board_size])
y = tf.placeholder(tf.float32, [None, board_size * board_size])

# Dropout placeholder
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    return tf.nn.conv2d(padded, W, strides=[1, 1, 1, 1], padding="VALID")

def maxpool2d(x):
    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(padded, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def cnn_forward(data):
    # Weights
    weights = {"W_conv1": tf.Variable(tf.random_normal([3, 3, 1, 64])),
    "W_conv2": tf.Variable(tf.random_normal([3, 3, 64, 64])),
    "W_conv3": tf.Variable(tf.random_normal([3, 3, 64, 128])),
    "W_conv4": tf.Variable(tf.random_normal([3, 3, 128, 128])),
    "W_conv5": tf.Variable(tf.random_normal([3, 3, 128, 256])),
    "W_conv6": tf.Variable(tf.random_normal([3, 3, 256, 256])),
    "W_fc": tf.Variable(tf.random_normal([6*6*256, 2048])),
    "out": tf.Variable(tf.random_normal([2048, n_classes]))}

    # Biases
    biases = {"b_conv1": tf.Variable(tf.random_normal([64])),
    "b_conv2": tf.Variable(tf.random_normal([64])),
    "b_conv3": tf.Variable(tf.random_normal([128])),
    "b_conv4": tf.Variable(tf.random_normal([128])),
    "b_conv5": tf.Variable(tf.random_normal([256])),
    "b_conv6": tf.Variable(tf.random_normal([256])),
    "b_fc": tf.Variable(tf.random_normal([2048])),
    "b_fc2": tf.Variable(tf.random_normal([2048])),
    "out": tf.Variable(tf.random_normal([n_classes]))}

    data = tf.reshape(data, shape=[-1, 19, 19, 1])

    # Forward prop
    conv1 = conv2d(data, weights["W_conv1"]) + biases["b_conv1"]
    conv1 = tf.nn.relu(conv1)

    conv2 = conv2d(conv1, weights["W_conv2"]) + biases["b_conv2"]
    conv2 = tf.nn.relu(conv2)

    conv3 = conv2d(conv2, weights["W_conv3"]) + biases["b_conv3"]
    conv3 = tf.nn.relu(conv3)

    conv4 = conv2d(conv3, weights["W_conv4"]) + biases["b_conv4"]
    conv4 = tf.nn.relu(conv4)
    conv4 = maxpool2d(conv4)

    conv5 = conv2d(conv4, weights["W_conv5"]) + biases["b_conv5"]
    conv5 = tf.nn.relu(conv5)

    conv6 = conv2d(conv5, weights["W_conv6"]) + biases["b_conv6"]
    conv6 = tf.nn.relu(conv6)
    conv6 = maxpool2d(conv6)

    fc = tf.reshape(conv6, [-1, 6*6*256])
    fc = tf.nn.relu(tf.matmul(fc, weights["W_fc"]) + biases["b_fc"])

    # Dropout
    fc_drop = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc_drop, weights["out"]) + biases["out"]

    return output

def cnn2_forward(data):
    weight_scale = 0.1

    # Weights
    weights = {"W_conv1": tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=weight_scale)),
    "W_conv2": tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=weight_scale)),
    "W_fc": tf.Variable(tf.random_normal([10*10*32, 256], stddev=weight_scale)),
    "out": tf.Variable(tf.random_normal([256, n_classes], stddev=weight_scale))}

    # Biases
    biases = {"b_conv1": tf.Variable(tf.random_normal([32], stddev=weight_scale)),
    "b_conv2": tf.Variable(tf.random_normal([32], stddev=weight_scale)),
    "b_fc": tf.Variable(tf.random_normal([256], stddev=weight_scale)),
    "out": tf.Variable(tf.random_normal([n_classes], stddev=weight_scale))}

    data = tf.reshape(data, shape=[-1, 19, 19, 1])
    # Forward prop
    conv1 = conv2d(data, weights["W_conv1"]) + biases["b_conv1"]
    conv1 = tf.nn.relu(conv1)

    conv2 = conv2d(conv1, weights["W_conv2"]) + biases["b_conv2"]
    conv2 = tf.nn.relu(conv2)
    conv2 = maxpool2d(conv2)

    # Dropout
    conv2_drop = tf.nn.dropout(conv2, keep_prob * 1.5)

    fc = tf.reshape(conv2_drop, [-1, 10*10*32])
    fc = tf.matmul(fc, weights["W_fc"]) + biases["b_fc"]
    fc = tf.nn.relu(fc)

    fc_drop = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc_drop, weights["out"]) + biases["out"]
    output = tf.nn.sigmoid(output)

    return output

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
    if mode == "cnn":
        pred = cnn_forward(x)
    else:
        pred = nn_forward(x)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    saver.restore(sess=sess, save_path=save_path)
    return {"session": sess, "prediction": pred}

# Changes the player turn by changing 1s to 2s and vice versa
def switch_player_perspec(arry):
    if arry == global_vars_go.bot_in:
        return global_vars_go.player_in
    elif arry == global_vars_go.player_in:
        return global_vars_go.bot_in
    else:
        return 0

def train_neural_network(x, gameData, nnType="cnn"):
    # Run a different model based on user input
    if nnType == "cnn":
        print("Training with a cnn")
        prediction = cnn_forward(x)
    elif nnType == "cnn2":
        print("Training with a cnn2")
        prediction = cnn2_forward(x)
    else: # Normal neural network
        print("Training with a standard nn")
        prediction = nn_forward(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()
    save_path = "checkpoints/next_move_model.ckpt"

    test_split = 100 # How many games are reserved for testing
    train_data = gameData[:-test_split]
    test_data = gameData[-test_split:]

    hm_epochs = 100
    hm_batches = int(len(train_data)/batch_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            batch_loss = 0
            batch_index = 0
            batch_display_index = 0
            train_boards = []#np.zeros(batch_size * board_size * board_size).reshape(batch_size, board_size * board_size)
            train_next_moves = []#np.zeros(batch_size * board_size * board_size).reshape(batch_size, board_size * board_size)
            for game_index in range(len(train_data)):
                board = setup_board(train_data[game_index])
                for node in train_data[game_index].get_main_sequence():
                    vfunc = np.vectorize(switch_player_perspec)
                    board = vfunc(board) # Changes player perspective, black becomes white and vice versa
                    node_move = node.get_move()
                    if node_move[1] is not None:
                        train_boards.append(np.copy(board))
                        next_move = np.zeros(board_size * board_size).reshape(board_size, board_size)
                        next_move[node_move[1][0], node_move[1][1]] = global_vars_go.bot_in # y = an array in the form [board_x_position, board_y_position]
                        train_next_moves.append(next_move)

                        # Debugging - printing and board graph
                        # print(train_boards[-1])
                        # print(train_next_moves[-1].astype(int))
                        # plt.imshow(train_boards[-1] + train_next_moves[-1]*3)
                        # ax = plt.gca();
                        # ax.grid(color='black', linestyle='-', linewidth=1)
                        # ax.set_xticks(np.arange(0, 19, 1));
                        # ax.set_yticks(np.arange(0, 19, 1));
                        # ax.set_xticklabels(np.arange(0, 19, 1));
                        # ax.set_yticklabels(np.arange(0, 19, 1));
                        # plt.show()

                        board = go_board.make_move(board, node_move[1], global_vars_go.bot_in, global_vars_go.player_in) # Update board with new move
                        if board is None:
                            print("ERROR! Illegal move, {}, while training".format(node_move[1]))
                    if len(train_boards) >= batch_size: # Send chunk to GPU at batch limit
                        _, c = sess.run([optimizer, cost], feed_dict = {x: np.array(train_boards).reshape(-1, board_size * board_size), y: np.array(train_next_moves).reshape(-1, board_size * board_size), keep_prob: 0.5}) # TODO: Make sure array keeps shape
                        epoch_loss += c
                        batch_loss += c
                        train_boards = []
                        train_next_moves = []
                        batch_index += 1
                        if batch_index >= batch_display_stride:
                            batch_index = 0
                            batch_display_index += 1
                            print("Epoch {}, Batch {} completed, Loss: {}".format(epoch+1, batch_display_index, batch_loss))
                            batch_loss = 0

            # Finish of what is remaining in the batch and give a visual update
            _, c = sess.run([optimizer, cost], feed_dict = {x: np.array(train_boards).reshape(-1, board_size * board_size), y: np.array(train_next_moves).reshape(-1, board_size * board_size), keep_prob: 0.5}) # TODO: Make sure array keeps shape
            epoch_loss += c

            saver.save(sess=sess, save_path=("checkpoints/nm_epoch_" + str(epoch+1) + ".ckpt"))
            print("\nEpoch", epoch+1, "completed out of", hm_epochs, ", Loss:", epoch_loss, "Accuracy:", test_accuracy(test_data, {"session": sess, "prediction": prediction}),"\n")
        saver.save(sess=sess, save_path=save_path)

def train(gameData, nnType):
    train_neural_network(x, gameData, nnType)

def get_prob_board(boards, model):
    boards = boards.reshape(-1, board_size * board_size)
    move = model["session"].run(model["prediction"], feed_dict = {x: boards, keep_prob: 1.0})
    return move

def predict_move(board, model, level=0):
    c_board = np.copy(board.reshape(board_size * board_size))
    prob_board = get_prob_board(c_board, model).reshape(board_size * board_size)
    sorted_board = np.asarray(sorted(enumerate(prob_board), reverse = True, key=lambda i:i[1]))

    move_found = False
    i = 0
    while move_found == False:
        if i >= len(c_board):
            move_found = True
            return(-1)
        if c_board[int(sorted_board[i][0])] == 0 and go_board.make_move(c_board.reshape(board_size, board_size), np.array([int(sorted_board[i][0]/board_size), int(sorted_board[i][0] % board_size)]), global_vars_go.bot_in, global_vars_go.player_in, debug=False) is not None:
            c_board[int(sorted_board[i][0])] = 1
            move_found = True
            return np.array([int(sorted_board[i][0]/board_size), int(sorted_board[i][0] % board_size)])
        i += 1

def test_accuracy(gameData, model):
    total = 0
    correct = 0
    train_actual_moves = []
    train_boards = []
    for game_index in range(len(gameData)): # Relative index of game to batch
        board = setup_board(gameData[game_index])
        for node in gameData[game_index].get_main_sequence():
            vfunc = np.vectorize(switch_player_perspec)
            board = vfunc(board) # Changes player perspective, black becomes white and vice versa
            node_move = node.get_move()
            if node_move[1] is not None:
                train_boards.append(np.copy(board))
                train_actual_moves.append([node_move[1][0], node_move[1][1]])
                board = go_board.make_move(board, node_move[1], global_vars_go.bot_in, global_vars_go.player_in) # Update board with new move
                if board is None:
                    print("ERROR! Illegal move, {}, while training".format(node_move[1]))
            if len(train_actual_moves) >= batch_size: # Send chunk to GPU at batch limit
                prob_boards = get_prob_board(np.array(train_boards), model).reshape((-1, board_size, board_size))
                train_predicted_moves = []
                for pb in prob_boards:
                    unrav_tuple = np.unravel_index(pb.argmax(), pb.shape)
                    train_predicted_moves.append([unrav_tuple[0], unrav_tuple[1]])
                for i in range(len(train_actual_moves)): # Test the accuracy of the batch
                    if train_actual_moves[i][0] == train_predicted_moves[i][0] and train_actual_moves[i][1] == train_predicted_moves[i][1]:
                        correct += 1
                    total += 1
                train_boards = []
                train_actual_moves = []

    for i in range(len(train_actual_moves)): # Test the accuracy of the leftover batch
        if train_actual_moves[i][0] == train_predicted_moves[i][0] and train_actual_moves[i][1] == train_predicted_moves[i][1]:
            correct += 1
        total += 1

    return correct/total

def setup_board(game):
    preboard, plays = sgf_moves.get_setup_and_moves(game)
    rpreboard = preboard.board
    board = np.zeros((board_size, board_size))
    if len(plays) < 1: # Return an empty board if the game has no moves
        return board
    if plays[0][0] == "b":
        color_stone = global_vars_go.bot_in
    else:
        color_stone = global_vars_go.player_in
    for i in range(len(rpreboard)):
        for j in range(len(rpreboard[i])):
            if rpreboard[i][j] == "b":
                board[i][j] = color_stone

    return board.astype(int)
