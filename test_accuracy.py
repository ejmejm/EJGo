import glob
import os
import sys
from sgfmill.sgfmill import sgf
import go_nn as go_learn
import global_vars_go as gvg

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

model = go_learn.setup_model(cont_save=True)

print("Begin testing...")
print("Accuracy:", go_learn.test_accuracy(games, model))
print("Finished testing")
