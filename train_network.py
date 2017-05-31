import glob
import os
import sys
from sgfmill.sgfmill import sgf
import go_nn as go_learn

kifuPath = "./kifu"
games = []
num_games = 5000
nnType = "nn"
if len(sys.argv) >= 2:
    num_games = int(sys.argv[1])
if len(sys.argv) >= 3:
	nnType = sys.argv[2]

print("Loading game data...")

i = 0
for filename in glob.glob(os.path.join(kifuPath, "*.sgf")):
    if i < num_games:
        with open(filename, "rb") as f:
            games.append(sgf.Sgf_game.from_bytes(f.read()))
            i += 1

print("Done loading", len(games), "games")

print("Begin training...")
go_learn.train(games, nnType)
print("Finished training")
