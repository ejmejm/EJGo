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
                train_boards.append(np.copy(board))

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
                        {"target": train_next_moves[int(-len(train_boards)*vs):]}), n_epoch=2,
                        batch_size=gvg.train_batch_size, snapshot_step=5000, show_metric=True)
                train_boards = []
                train_next_moves = []
                # if batch_index >= batch_display_stride:
                #     batch_index = 0
                #     batch_display_index += 1
                #     batch_acc /= batch_display_stride
                #     print("Epoch {}, Batch {} completed, Loss: {}, Accuracy: {}".format(epoch+1, batch_display_index, batch_loss, batch_acc))
                #     batch_loss = 0
                #     batch_acc = 0

    # Finish of what is remaining in the batch and give a visual update
    model.fit({"input": train_boards[:int(-len(train_boards)*vs)]},
            {"target": train_next_moves[:int(-len(train_boards)*vs)]},
            validation_set=({"input": train_boards[int(-len(train_boards)*vs):]},
            {"target": train_next_moves[int(-len(train_boards)*vs):]}), n_epoch=2,
            batch_size=gvg.train_batch_size, snapshot_epoch=True, show_metric=True)

    #model.save("test.tflearn")

def predict_move(board, model, level=0):
    return nanargmax(np.array(model.predict(board.reshape(-1, gvg.board_size, gvg.board_size, gvg.board_channels))).reshape(gvg.board_size, gvg.board_size))

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
