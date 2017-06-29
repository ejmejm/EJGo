import glob
import os
import sys
from sgfmill.sgfmill import sgf
import global_vars_go as gvg
import loader
import utils
import board3d as go_board
import numpy as np

kifuPath = "./kifu"
games = []
num_games = gvg.num_games
from_game = gvg.from_test_games

print("Loading game data...")

i = 0
for filename in glob.glob(os.path.join(kifuPath, "*.sgf")):
    if from_game <= i < from_game+num_games:
        with open(filename, "rb") as f:
            games.append(sgf.Sgf_game.from_bytes(f.read()))
    i += 1

print("Done loading {} games".format(len(games)))

model = loader.load_model_from_file(gvg.nn_type)
#X, Y = utils.games_to_states(games)

train_boards = []
train_next_moves = []
for game_index in range(len(games)):
    board = go_board.setup_board(games[game_index])
    for node in games[game_index].get_main_sequence():
        board = go_board.switch_player_perspec(board) # Changes player perspective, black becomes white and vice versa

        node_move = node.get_move()[1]
        if node_move is not None:
            train_boards.append(np.copy(board))
            next_move = np.zeros(gvg.board_size * gvg.board_size).reshape(gvg.board_size, gvg.board_size)
            next_move[node_move[0], node_move[1]] = gvg.filled # y = an array in the form [board_x_position, board_y_position]
            train_next_moves.append(next_move.reshape(gvg.board_size * gvg.board_size))

            board = go_board.make_move(board, node_move, gvg.bot_channel, gvg.player_channel) # Update board with new move
            if board is None:
                print("ERROR! Illegal move, {}, while training".format(node_move))

print("Begin testing...")
correct = 0
for i in range(len(train_boards)):
    enc_board = go_board.get_encoded_board(np.copy(train_boards[i])).reshape(1, gvg.board_size, gvg.board_size, 20)
    pred = np.asarray(model.predict(enc_board)).reshape(gvg.board_size * gvg.board_size)
    if pred.argmax() == train_next_moves[i].argmax():
        correct += 1
print("Accuracy: {}".format(correct/len(train_boards)))
print("Finished testing")
