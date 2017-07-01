import numpy as np
import random
import os
import sys
from Engine import *
from GTP import Move
import go_nn as go_learn
import board3d as go_board

class TFEngine(BaseEngine):
    def __init__(self, eng_name, model):
        super(TFEngine,self).__init__()
        self.eng_name = eng_name
        self.model = model

        self.last_move_probs = np.zeros((gvg.board_size, gvg.board_size,))
        self.kibitz_mode = False

    def name(self):
        return self.eng_name

    def version(self):
        return "1.0"

    def set_board_size(self, N):
        if N != gvg.board_size:
            return False
        return BaseEngine.set_board_size(self, N)

    # def pick_book_move(self, color):
    #     if self.book:
    #         book_move = Book.get_book_move(self.board, self.book)
    #         if book_move:
    #             print "playing book move", book_move
    #             return Move(book_move[0], book_move[1])
    #         print "no book move"
    #     else:
    #         print "no book"
    #     return None

    def pick_model_move(self, color):
        # If asked to make a move for enemy player, switch perspective as if we are them
        if channel_from_color(color) == gvg.player_channel:
            self.board = go_board.switch_player_perspec(self.board)

        enc_board = go_board.get_encoded_board(self.board)
        pred = np.asarray(self.model.predict(enc_board. \
        reshape(1, gvg.board_size, gvg.board_size, gvg.enc_board_channels))). \
        reshape(gvg.board_size * gvg.board_size)
        prob_board = np.array(self.model.predict(go_board.get_encoded_board(self.board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)).reshape(1, 19, 19, 20))).reshape((gvg.board_size, gvg.board_size))
        self.last_move_probs = prob_board

        move = go_learn.predict_move(self.board, self.model, prob_board=prob_board)

        return Move(move[0], move[1])

    def pick_move(self, color):
        #if self.opponent_passed:
        #    return Move.Pass

        return self.pick_model_move(color)

    def get_last_move_probs(self):
        return self.last_move_probs

    def stone_played(self, x, y, color):
        # if we are in kibitz mode, we want to compute model probabilities for ALL turns
        # if self.kibitz_mode:
        #     self.pick_model_move(color)
        #     true_stderr.write("probability of played move %s (%d, %d) was %.2f%%\n" % (color_names[color], x, y, 100*self.last_move_probs[x,y]))

        BaseEngine.stone_played(self, x, y, color)

    def toggle_kibitz_mode(self):
        self.kibitz_mode = ~self.kibitz_mode
        return self.kibitz_mode
