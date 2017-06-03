import sys

empty_in = 0
bot_in = 1
player_in = 2
num_games = 1000
hm_epochs = 50
train_batch_size = 128
train_display_stride = 400
file_load_split = 15000
nn_type = "cnn2"
validation_split = 0.03 # What fraction of games are reserved for validation
from_test_games = 100000

if len(sys.argv) >= 2:
    num_games = int(sys.argv[1])
if len(sys.argv) >= 3:
	nn_type = sys.argv[2]
