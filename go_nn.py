import tensorflow as tf
import numpy as np
import time
import board3d as go_board
import matplotlib.pyplot as plt
import global_vars_go as gvg
import models.cnn

batch_size = gvg.process_batch_size # How many board states (not full games) to send to GPU at once, about 200 is the limit of this GPU's RAM

def train_network(game_data, model):
    vs = gvg.validation_split # What fraction of games are reserved for validation

    hm_batches = int(len(game_data)/batch_size)

    train_boards = []
    train_next_moves = []
    for game_index in range(len(game_data)):
        board = go_board.setup_board(game_data[game_index])
        for node in game_data[game_index].get_main_sequence():
            board = go_board.switch_player_perspec(board) # Changes player perspective, black becomes white and vice versa

            node_move = node.get_move()[1]
            if node_move is not None:
                train_boards.append(go_board.get_encoded_board(np.copy(board)))

                next_move = np.zeros(gvg.board_size * gvg.board_size).reshape(gvg.board_size, gvg.board_size)
                next_move[node_move[0], node_move[1]] = gvg.filled # y = an array in the form [board_x_position, board_y_position]
                train_next_moves.append(next_move.reshape(gvg.board_size * gvg.board_size))

                board = go_board.make_move(board, node_move, gvg.bot_channel, gvg.player_channel) # Update board with new move
                if board is None:
                    print("ERROR! Illegal move, {}, while training".format(node_move[1]))
            if len(train_boards) >= batch_size: # Send chunk to GPU at batch limit
                model.fit({"input": train_boards[:int(-len(train_boards)*vs)]},
                        {"target": train_next_moves[:int(-len(train_boards)*vs)]},
                        validation_set=({"input": train_boards[int(-len(train_boards)*vs):]},
                        {"target": train_next_moves[int(-len(train_boards)*vs):]}), n_epoch=1,
                        batch_size=gvg.train_batch_size, snapshot_step=7500, show_metric=True)
                train_boards = []
                train_next_moves = []

    # Finish of what is remaining in the batch and give a visual update
    model.fit({"input": train_boards[:int(-len(train_boards)*vs)]},
            {"target": train_next_moves[:int(-len(train_boards)*vs)]},
            validation_set=({"input": train_boards[int(-len(train_boards)*vs):]},
            {"target": train_next_moves[int(-len(train_boards)*vs):]}), n_epoch=1,
            batch_size=gvg.train_batch_size, snapshot_step=7500, show_metric=True)

    #model.save("test.tflearn")

def predict_move(orig_board, model, level=0, prob_board=None):
    board = go_board.get_encoded_board(np.copy(orig_board))
    if prob_board is None:
        prob_board = np.array(model.predict(board)).reshape(gvg.board_size, gvg.board_size)

    found_move = False
    while found_move == False:
        move = nanargmax(prob_board)
        if board[move[0]][move[1]][gvg.player_channel] == gvg.filled or board[move[0]][move[1]][gvg.bot_channel] == gvg.filled or \
        go_board.legal_move(orig_board, move, move_made=False, player=gvg.bot_channel) == False:
            prob_board[move[0]][move[1]] = -999999.0
        else:
            found_move = True

    return move

# Source: https://stackoverflow.com/questions/21989513/finding-index-of-maximum-value-in-array-with-numpy
def nanargmax(a):
    idx = np.argmax(a, axis=None)
    multi_idx = np.unravel_index(idx, a.shape)
    if np.isnan(a[multi_idx]):
        nan_count = np.sum(np.isnan(a))
        # In numpy < 1.8 use idx = np.argsort(a, axis=None)[-nan_count-1]
        idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
        multi_idx = np.unravel_index(idx, a.shape)
    return multi_idx
