import sys
import numpy as np
import go_nn as go_learn

if len(sys.argv) == 2:
    #when passing in board as an argument, -1 is the player, 0 is empty, and 1 is the bot
    board = np.fromstring(str(sys.argv[1]), dtype=int, sep=',')

    prob_board = go_learn.get_move(board).reshape(go_learn.board_size, go_learn.board_size)
    sorted_index = [i[0] for i in sorted(enumerate(prob_board), key=lambda x:x[1], reverse=True)]
else:
    print("ERROR! You need to include the current board and only that as an argument")
