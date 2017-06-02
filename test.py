import glob
import os
import numpy as np
from sgfmill.sgfmill import sgf
from sgfmill.sgfmill import sgf_moves
import go_nn as go_learn
import tensorflow as tf
import global_vars_go

kifuPath = "./kifu"

board_size = 19

print("Loading game data...")

i = 0
for filename in glob.glob(os.path.join(kifuPath, "*.sgf")):
    if i < 1:
        with open("./kifu/2007-07-29-15.sgf", "rb") as f:
            game = sgf.Sgf_game.from_bytes(f.read())
            i += 1

print("Done loading games")

def setup_board(game):
    preboard, plays = sgf_moves.get_setup_and_moves(game)
    rpreboard = preboard.board
    board = np.zeros((board_size, board_size))
    if plays[0][0] == "b":
        color_stone = global_vars_go.bot_in
    else:
        color_stone = global_vars_go.player_in
    for i in range(len(rpreboard)):
        for j in range(len(rpreboard[i])):
            if rpreboard[i][j] == "b":
                board[i][j] = color_stone

    return board.astype(int)

board = setup_board(game)
print(board)
print("test")

# winner = games[0].get_winner()
# board_size = games[0].get_size()
# root_node = games[0].get_root()
# b_player = root_node.get("PB")
# w_player = root_node.get("PW")
#
# for node in games[0].get_main_sequence():
#     if node.get_move()[1] != None:
#         x = node.get_move()[1]
#         #print(node.get_move())
#
# model = go_learn.load("checkpoints/next_move_model.ckpt")
# print(model["session"])
# b = np.zeros(19*19).reshape(1, board_size * board_size)
# #print(go_learn.get_prob_board(b, model))
# print(go_learn.predict_move(b, model))
#print(go_learn.get_prob_board(np.zeros(19*19)))

#print(go_learn.get_move(np.zeros(19*19)).reshape(go_learn.board_size, go_learn.board_size).astype(int))

#print("Begin learning...")
#go_learn.train(games)
#go_learn.load("checkpoints/next_move_model")
