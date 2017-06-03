import sys

empty_in = 0 # Board integer for empty spaces
bot_in = 1 # Board integer for spaces occupied by the bot
player_in = 2 # Board integer for spaces occupied by the player
num_games = 1000 # Number of games to train and test accuracy on
hm_epochs = 50 # Number of loops through all training data
train_batch_size = 128 # Number of board states to send to the GPU at once
train_display_stride = 400 # Number of batches before giving a visual update
file_load_split = 30000 # Number of games to load from disk to RAM at once
nn_type = "cnn2" # Which model to use for training and testing
validation_split = 0.03 # What fraction of games are reserved for validation
from_test_games = 100000 # How many games are reserved for training/where testing starts

if len(sys.argv) >= 2:
    num_games = int(sys.argv[1])
if len(sys.argv) >= 3:
	nn_type = sys.argv[2]
