import glob
import os
from sgfmill.sgfmill import sgf
import go_nn as go_learn

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

print("Begin learning...")
go_learn.train(games)
