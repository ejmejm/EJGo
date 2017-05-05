import os
import numpy as np
from sgfmill.sgfmill import sgf
import go_nn as go_learn

# The bot is 1, the player is -1, and empty is 0
board = np.zeros((go_learn.board_size, go_learn.board_size))
model = go_learn.load("checkpoints/next_move_model.ckpt")

while True:
    player_input = input("Enter your move: ")
    player_move = [int(n) for n in player_input.split()]
    player_move[0] -= 1 # Change from 1-19 to 0-18
    player_move[1] -= 1
    if len(player_move) != 2:
        print("PLease enter 2 numbers")
    elif player_move[0] < 0 or player_move[0] > 18 or player_move[1] < 0 or player_move[1] > 18:
        print("That move was out of range")
    elif board[player_move[0], player_move[1]] == -1:
        print("That spot is already occupied by the player")
    elif board[player_move[0], player_move[1]] == 1:
        print("That spot is already occupied by the bot")
    else:
        board[player_move[0], player_move[1]] = -1
        bot_move = go_learn.predict_move(board, model)
        board[bot_move[0], bot_move[1]] = 1
        print("Bot move: ", bot_move)
