from GTP import Move
import copy
import board3d as go_board
import global_vars_go as gvg

class BaseEngine(object):
    def __init__(self):
        self.board = None
        self.opponent_passed = False
        self.state_stack = []

    def push_state(self):
        self.state_stack.append(copy.deepcopy(self.board))

    def pop_state(self):
        self.board = self.state_stack.pop()
        self.opponent_passed = False

    def undo(self):
        if len(self.state_stack) > 0:
            self.pop_state()
            print("BaseEngine: after undo, board is")
            self.board.show()
        else:
            print("BaseEngine: undo called, but state_stack is empty. Board is")
            self.board.show()

    # subclasses must override this
    def name(self):
        assert False

    # subclasses must override this
    def version(self):
        assert False

    # subclasses may override to only accept
    # certain board sizes. They should call this
    # base method.
    def set_board_size(self, N):
        if N != gvg.board_size:
            print("ERROR! Board size was set to {}, but it can only be {}".format(N, gvg.board_size))
        self.board = go_board.empty_board()
        return True

    def clear_board(self):
        self.board = go_board.empty_board()
        self.state_stack = []
        self.opponent_passed = False

    def set_komi(self, komi):
        self.komi = float(komi)

    def player_passed(self, color):
        self.push_state()
        #self.board.play_pass() TODO: create pass channel for board
        self.opponent_passed = True

    def stone_played(self, x, y, color):
        self.push_state()
        if board.color_to_play == gvg.kgs_black:
            go_board.make_move(self.board, [x-1, y-1], gvg.bot_channel, gvg.player_channel)
        else:
            go_board.make_move(self.board, [x-1, y-1], gvg.player_channel, gvg.bot_channel)
        self.opponent_passed = False
        #self.board.show()

    def move_was_played(self, move):
        if move.is_play():
            self.stone_played(move.x, move.y, self.board.color_to_play)
        elif move.is_pass():
            self.player_passed(self.board.color_to_play)

    # subclasses must override this
    def pick_move(self, color):
        assert False

    def generate_move(self, color, cleanup=False):
        move = self.pick_move(color)
        self.push_state()
        if move.is_play():
            self.board.play_stone(move.x, move.y, color)
        #self.board.show()
        return move

    def quit(self):
        pass

    def supports_final_status_list(self):
        return False


class IdiotEngine(BaseEngine):
    def __init__(self):
        super(IdiotEngine,self).__init__()

    def name(self):
        return "IdiotEngine"

    def version(self):
        return "1.0"

    def pick_move(self, color):
        for x in xrange(self.board.N):
            for y in xrange(self.board.N):
                if self.board.play_is_legal(x, y, color):
                    return Move(x,y)
        return Move.Pass()
