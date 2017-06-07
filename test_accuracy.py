import glob
import os
import sys
from sgfmill.sgfmill import sgf
import global_vars_go as gvg
import loader
import utils

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
X, Y = utils.games_to_states(games)

print("Begin testing...")
print("Accuracy:", model.evaluate(X, Y))
print("Finished testing")
