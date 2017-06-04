import tensorflow as tf
import numpy as np
import time
import board as go_board
import global_vars_go
import matplotlib.pyplot as plt
from sgfmill.sgfmill import sgf_moves
import global_vars_go as gvg

board_size = 19

n_nodes_hl1 = 300
n_nodes_hl2 = 300
n_nodes_hl3 = 300

batch_size = gvg.train_batch_size # How many board states (not full games) to send to GPU at once, about 200 is the limit of this GPU's RAM
batch_display_stride = gvg.train_display_stride # How many batches to send to GPU before displaying a visual update

n_classes = board_size * board_size

# IO placeholders
x = tf.placeholder(tf.float32, [None, board_size * board_size])
y = tf.placeholder(tf.float32, [None, board_size * board_size])

# Dropout placeholder
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

def conv2dp1(x, W):
    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    return tf.nn.conv2d(padded, W, strides=[1, 1, 1, 1], padding="VALID")

def conv2dp2(x, W):
    padded = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")
    return tf.nn.conv2d(padded, W, strides=[1, 1, 1, 1], padding="VALID")

def conv2dp3(x, W):
    padded = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], "CONSTANT")
    return tf.nn.conv2d(padded, W, strides=[1, 1, 1, 1], padding="VALID")

def maxpool2d(x):
    padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(padded, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

def cnn_forward(data):
    weight_scale = 0.1

    # Weights
    weights = {"W_conv1": tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=weight_scale)),
    "W_conv2": tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=weight_scale)),
    "W_conv3": tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=weight_scale)),
    "W_conv4": tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=weight_scale)),
    "W_conv5": tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=weight_scale)),
    "W_conv6": tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=weight_scale)),
    "W_conv7": tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=weight_scale)),
    "W_conv8": tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=weight_scale)),
    "W_conv9": tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=weight_scale)),
    "W_fc": tf.Variable(tf.random_normal([6*6*256, 1024], stddev=weight_scale)),
    "out": tf.Variable(tf.random_normal([1024, n_classes], stddev=weight_scale))}

    # Biases
    biases = {"b_conv1": tf.Variable(tf.zeros([64])),
    "b_conv2": tf.Variable(tf.zeros([64])),
    "b_conv3": tf.Variable(tf.zeros([128])),
    "b_conv4": tf.Variable(tf.zeros([128])),
    "b_conv5": tf.Variable(tf.zeros([256])),
    "b_conv6": tf.Variable(tf.zeros([256])),
    "b_conv7": tf.Variable(tf.zeros([512])),
    "b_conv8": tf.Variable(tf.zeros([512])),
    "b_conv9": tf.Variable(tf.zeros([512])),
    "b_fc": tf.Variable(tf.zeros([1024])),
    "out": tf.Variable(tf.zeros([n_classes]))}

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

    conv7 = conv2d(conv6, weights["W_conv7"]) + biases["b_conv7"]
    conv7 = tf.nn.relu(conv7)

    conv8 = conv2d(conv7, weights["W_conv8"]) + biases["b_conv8"]
    conv8 = tf.nn.relu(conv8)

    conv9 = conv2d(conv8, weights["W_conv9"]) + biases["b_conv9"]
    conv9 = tf.nn.relu(conv9)

    conv9_drop = tf.nn.dropout(conv9, keep_prob)

    fc = tf.reshape(conv9_drop, [-1, 6*6*512])
    fc = tf.nn.relu(tf.matmul(fc, weights["W_fc"]) + biases["b_fc"])

    # Dropout
    fc_drop = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc_drop, weights["out"]) + biases["out"]
    output = tf.nn.sigmoid(output)

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

def cnn3_forward(data):
    weight_scale = 0.1

    # Weights
    weights = {"W_conv1": tf.Variable(tf.random_normal([7, 7, 1, 64], stddev=weight_scale)),
    "W_conv2": tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=weight_scale)),
    "W_conv3": tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=weight_scale)),
    "W_conv4": tf.Variable(tf.random_normal([5, 5, 64, 48], stddev=weight_scale)),
    "W_conv5": tf.Variable(tf.random_normal([5, 5, 48, 48], stddev=weight_scale)),
    "W_conv6": tf.Variable(tf.random_normal([5, 5, 48, 32], stddev=weight_scale)),
    "W_conv7": tf.Variable(tf.random_normal([5, 5, 32, 32], stddev=weight_scale)),
    "W_fc": tf.Variable(tf.random_normal([9*9*32, 1024], stddev=weight_scale)),
    "out": tf.Variable(tf.random_normal([1024, n_classes], stddev=weight_scale))}

    # Biases
    biases = {"b_conv1": tf.Variable(tf.zeros([64])),
    "b_conv2": tf.Variable(tf.zeros([64])),
    "b_conv3": tf.Variable(tf.zeros([64])),
    "b_conv4": tf.Variable(tf.zeros([48])),
    "b_conv5": tf.Variable(tf.zeros([48])),
    "b_conv6": tf.Variable(tf.zeros([32])),
    "b_conv7": tf.Variable(tf.zeros([32])),
    "b_fc": tf.Variable(tf.zeros([1024])),
    "out": tf.Variable(tf.zeros([n_classes]))}

    data = tf.reshape(data, shape=[-1, 19, 19, 1])
    # Forward prop
    conv1 = conv2d(data, weights["W_conv1"]) + biases["b_conv1"]
    conv1 = tf.nn.relu(conv1)

    conv2 = conv2d(conv1, weights["W_conv2"]) + biases["b_conv2"]
    conv2 = tf.nn.relu(conv2)

    conv3 = conv2dp1(conv2, weights["W_conv3"]) + biases["b_conv3"]
    conv3 = tf.nn.relu(conv3)

    conv4 = conv2dp1(conv3, weights["W_conv4"]) + biases["b_conv4"]
    conv4 = tf.nn.relu(conv4)

    conv5 = conv2dp1(conv4, weights["W_conv5"]) + biases["b_conv5"]
    conv5 = tf.nn.relu(conv5)

    conv6 = conv2dp1(conv5, weights["W_conv6"]) + biases["b_conv6"]
    conv6 = tf.nn.relu(conv6)

    conv7 = conv2dp1(conv6, weights["W_conv7"]) + biases["b_conv7"]
    conv7 = tf.nn.relu(conv7)

    conv7_drop = tf.nn.dropout(conv7, keep_prob)

    fc = tf.reshape(conv7_drop, [-1, 9*9*32])
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

# Depricated, instead use setup_model()
# def load(save_path):
#     sess = tf.Session()
#
#     if mode == "cnn":
#         pred = cnn_forward(x)
#     elif mode == "cnn2":
#         pred = cnn2_forward(x)
#     else:
#         pred = nn_forward(x)
#
#     saver = tf.train.Saver()
#
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess=sess, save_path=save_path)
#     return {"session": sess, "prediction": pred}

# Changes the player turn by changing 1s to 2s and vice versa
def switch_player_perspec(arry):
    if arry == global_vars_go.bot_in:
        return global_vars_go.player_in
    elif arry == global_vars_go.player_in:
        return global_vars_go.bot_in
    else:
        return 0

def train_neural_network(x, gameData, model, epoch, hm_epochs):
    validation_split = gvg.validation_split # What fraction of games are reserved for validation
    train_data = gameData[:-int(validation_split*len(gameData))]
    validation_data = gameData[-int(validation_split*len(gameData)):]

    hm_batches = int(len(train_data)/batch_size)

    epoch_loss = 0
    batch_loss = 0
    batch_index = 0
    batch_display_index = 0
    train_boards = []
    train_next_moves = []
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
                _, c = model["session"].run([model["optimizer"], model["cost"]], feed_dict = {x: np.array(train_boards).reshape(-1, board_size * board_size), y: np.array(train_next_moves).reshape(-1, board_size * board_size), keep_prob: 0.5})
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
    _, c = model["session"].run([model["optimizer"], model["cost"]], feed_dict = {x: np.array(train_boards).reshape(-1, board_size * board_size), y: np.array(train_next_moves).reshape(-1, board_size * board_size), keep_prob: 0.5}) # TODO: Make sure array keeps shape
    epoch_loss += c

    model["saver"].save(sess=model["session"], save_path=("checkpoints/nm_epoch_" + str(epoch+1) + ".ckpt"))
    print("\nFile batch completed,", "Loss:", epoch_loss, "Accuracy:", test_accuracy(validation_data, model))
    model["saver"].save(sess=model["session"], save_path=model["save_path"])

def train(gameData, model, epoch, hm_epochs):
    train_neural_network(x, gameData, model, epoch, hm_epochs)

# Call before training to initialize the session and training variables
def setup_model(cont_save=False):
    session = tf.Session()

    if gvg.nn_type == "cnn":
        print("\nTraining with a cnn\n")
        prediction = cnn_forward(x)
    elif gvg.nn_type == "cnn2":
        print("\nTraining with a cnn2\n")
        prediction = cnn2_forward(x)
    elif gvg.nn_type == "cnn3":
        print("\nTraining with a cnn3\n")
        prediction = cnn3_forward(x)
    else:
        print("\nTraining with a standard nn\n")
        prediction = nn_forward(x)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    optimizer = tf.train.AdamOptimizer(learning_rate=gvg.learning_rate).minimize(cost)
    saver = tf.train.Saver()
    save_path = "checkpoints/next_move_model.ckpt"

    if cont_save == False:
        session.run(tf.global_variables_initializer())
    else:
        saver.restore(session, save_path)

    # Returns the entire model
    return {"session": session, "prediction": prediction, "cost": cost, "optimizer": optimizer, "saver": saver, "save_path": save_path}

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
