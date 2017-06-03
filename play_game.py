import os
import numpy as np
from sgfmill.sgfmill import sgf
import go_nn as go_learn
import board as go_board
import global_vars_go
import sys

# The bot is 1, the player is 2, and empty is 0

if len(sys.argv) >= 2:
    nnType = sys.argv[1]

go_learn.mode = nnType

board = np.zeros((go_learn.board_size, go_learn.board_size))
model = go_learn.load("checkpoints/next_move_model.ckpt")

bot = global_vars_go.bot_in
player = global_vars_go.player_in

def take_turn(player_move, set_stones=None):
    player_move[0] -= 1 # Change from 1-19 to 0-18
    player_move[1] -= 1
    if len(player_move) != 2:
        print("Please enter 2 numbers")
    elif player_move[0] < 0 or player_move[0] > 18 or player_move[1] < 0 or player_move[1] > 18:
        print("That move was out of range")
    elif board[player_move[0], player_move[1]] == player:
        print("That spot is already occupied by the player")
    elif board[player_move[0], player_move[1]] == bot:
        print("That spot is already occupied by the bot")
    else:
        go_board.make_move(board, player_move, player, bot)
        bot_move = go_learn.predict_move(board, model)
        go_board.make_move(board, bot_move, bot, player)
        print("Bot move: ", bot_move + 1)
        if set_stones != None:
            set_stones(board)
