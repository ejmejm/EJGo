import sys

empty = 0 # Board integer for empty spaces
filled = 1 # Board integer for filled spaces
bot_channel = 0 # Board channel used for bot moves
player_channel = 1 # Board channel used for bot moves
num_games = 1000 # Number of games to train and test accuracy on
hm_epochs = 50 # Number of loops through all training data
train_batch_size = 128 # Number of board states to send to the GPU at once
train_display_stride = 100 # Number of batches before giving a visual update
file_load_split = 30000 # Number of games to load from disk to RAM at once
nn_type = "cnn2" # Which model to use for training and testing
validation_split = 0.03 # What fraction of games are reserved for validation
from_test_games = 100000 # How many games are reserved for training/where testing starts
learning_rate = 0.001 # Learning rate for training
board_size = 19 # Side length of the Go board
board_channels = 2 # 3rd dimmension of the board

# Soon to be depricated
empty_in = 0 # Board integer for empty spaces
bot_in = 1 # Board integer for spaces occupied by the bot
player_in = -1 # Board integer for spaces occupied by the player