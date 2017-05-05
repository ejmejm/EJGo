import glob
import os
import numpy as np
from sgfmill.sgfmill import sgf
import go_nn as go_learn
import tensorflow as tf

kifuPath = "./kifu"
games = []

print("Loading game data...")

i = 0
for filename in glob.glob(os.path.join(kifuPath, "*.sgf")):
    if i < 30:
        with open(filename, "rb") as f:
            games.append(sgf.Sgf_game.from_bytes(f.read()))
            i += 1

print("Done loading games")

winner = games[0].get_winner()
board_size = games[0].get_size()
root_node = games[0].get_root()
b_player = root_node.get("PB")
w_player = root_node.get("PW")

for node in games[0].get_main_sequence():
    if node.get_move()[1] != None:
        x = node.get_move()[1]
        #print(node.get_move())

model = go_learn.load("checkpoints/next_move_model.ckpt")
print(model["session"])
b = np.zeros(19*19).reshape(1, board_size * board_size)
#print(go_learn.get_prob_board(b, model))
print(go_learn.predict_move(b, model))
#print(go_learn.get_prob_board(np.zeros(19*19)))

#print(go_learn.get_move(np.zeros(19*19)).reshape(go_learn.board_size, go_learn.board_size).astype(int))

#print("Begin learning...")
#go_learn.train(games)
#go_learn.load("checkpoints/next_move_model")
