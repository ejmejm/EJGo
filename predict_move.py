import sys
import numpy as np
import go_nn as go_learn

if len(sys.argv) == 2:
    #when passing in board as an argument, -1 is the player, 0 is empty, and 1 is the bot
    board = np.fromstring(str(sys.argv[1]), dtype=int, sep=',')

    prob_board = go_learn.get_move(board).reshape(go_learn.board_size* go_learn.board_size)
    sorted_board = sorted(enumerate(prob_board), reverse = True, key=lambda i:i[1])

    move_found = False
    i = 0
    while move_found == False:
        if i >= len(board):
            move_found = True
            print("Move: Pass")
        if board[sorted_board[i][0]] == 0:
            board[sorted_board[i][0]] = 1
            move_found = True
            print("Move: ", int(sorted_board[i][0]/go_learn.board_size) + 1, ", ", int(sorted_board[i][0] % go_learn.board_size) + 1)
        i += 1
else:
    print("ERROR! You need to include the current board and only that as an argument")
