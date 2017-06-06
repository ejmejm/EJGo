import glob
import os
import sys
import math
from sgfmill.sgfmill import sgf
import go_nn as go_learn
import global_vars_go as gvg
import loader

kifuPath = "./kifu"

file_load_split = gvg.file_load_split
num_games = gvg.num_games

print("Loading game data...")

load_batches = math.ceil(num_games/file_load_split)
hm_epochs = gvg.hm_epochs
model = loader.load_model(gvg.nn_type)

for epoch in range(hm_epochs):
    print("Beginning new epoch...")
    for lb in range(load_batches):
        games = []
        i = 0
        for filename in glob.glob(os.path.join(kifuPath, "*.sgf")):
            if lb*file_load_split <= i < (lb+1)*file_load_split and i < num_games:
                with open(filename, "rb") as f:
                    games.append(sgf.Sgf_game.from_bytes(f.read()))
            i += 1

        print("Done loading file bach of", len(games), "games")

        go_learn.train_network(games, model)
        print("\nFile batch", lb+1, "completed out of", load_batches, "\n")
    print("Epoch {} completed of {} epochs\n".format(epoch+1, hm_epochs))


print("Finished training")
