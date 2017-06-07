import os
import numpy as np
from sgfmill.sgfmill import sgf
import go_nn as go_learn
import board3d as go_board
import global_vars_go as gvg
import sys
import loader

board = np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels))
model = loader.load_model_from_file(gvg.nn_type)

def take_turn(player_move, set_stones=None):
    player_move[0] -= 1 # Change from 1-19 to 0-18
    player_move[1] -= 1
    if len(player_move) != 2:
        print("Please enter 2 numbers")
    elif player_move[0] < 0 or player_move[0] > 18 or player_move[1] < 0 or player_move[1] > 18:
        print("That move was out of range")
    elif board[player_move[0], player_move[1], gvg.player_channel] == gvg.filled:
        print("That spot is already occupied by the player")
    elif board[player_move[0], player_move[1], gvg.bot_channel] == gvg.filled:
        print("That spot is already occupied by the bot")
    else:
        go_board.make_move(board, player_move, gvg.player_channel, gvg.bot_channel)
        bot_move = np.asarray(go_learn.predict_move(board, model))
        if board[bot_move[0]][bot_move[1]][gvg.player_channel] == gvg.filled:
            print("Bot tried to move on player space")
        else:
            go_board.make_move(board, bot_move, gvg.bot_channel, gvg.player_channel)
            print("Bot move: ", bot_move + 1)
        if set_stones != None:
            set_stones(board)
