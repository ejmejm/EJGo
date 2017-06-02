import glob
import os
import sys
from sgfmill.sgfmill import sgf
import go_nn as go_learn

kifuPath = "./kifu"
games = []
num_games = 1000

print("Loading game data...")

i = 0
for filename in glob.glob(os.path.join(kifuPath, "*.sgf")):
    if i < num_games:
        with open(filename, "rb") as f:
            games.append(sgf.Sgf_game.from_bytes(f.read()))
            i += 1
games = games[950:]

print("Done loading games")

go_learn.mode = "cnn"
model = go_learn.load("checkpoints/next_move_model.ckpt")

print("Begin testing...")
print("Accuracy:", go_learn.test_accuracy(games, model)*100, "%")
print("Finished testing")
