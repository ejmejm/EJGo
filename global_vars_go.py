import sys

empty = 0 # Board integer for empty spaces
filled = 1 # Board integer for filled spaces
bot_channel = 0 # Board channel used for bot moves
player_channel = 1 # Board channel used for bot moves
num_games = 1000 # Number of games to train and test accuracy on
hm_epochs = 7 # Number of loops through all training data
process_batch_size = 64 * 500 # Number of games to process into board states before fitting
train_batch_size = 64 # Number of board states to send to the GPU at once
train_display_stride = 30 # Number of batches before giving a visual update
file_load_split = 30000 # Number of games to load from disk to RAM at once
nn_type = "cnn" # Which model to use for training and testing
validation_split = 0.03 # What fraction of games are reserved for validation
from_test_games = 100000 # How many games are reserved for training/where testing starts
learning_rate = 0.001 # Learning rate for training
board_size = 19 # Side length of the Go board
board_channels = 2 # 3rd dimmension of the board
checkpoint_path = "checkpoints/" # Where model checkpoints are stored
cont_from_save = "false" # True if loading a model save from a checkpoint and False otherwise
kgs_empty = 0 # Empty with KGS engine
kgs_black = 1 # Black stone with KGS engine
kgs_white = 2 # White stone with KGS engine
black_channel = bot_channel # Black stone with KGS engine
white_channel = player_channel # White stone with KGS engine

# Soon to be depricated
empty_in = 0 # Board integer for empty spaces
bot_in = 1 # Board integer for spaces occupied by the bot
player_in = 2 # Board integer for spaces occupied by the player

if len(sys.argv) >= 2:
    num_games = int(sys.argv[1])
if len(sys.argv) >= 3:
	nn_type = sys.argv[2]
if len(sys.argv) >= 4:
	cont_from_save = sys.argv[3]
