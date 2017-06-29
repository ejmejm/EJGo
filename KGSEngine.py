#Source: https://github.com/TheDuck314/go-NN

#!/usr/bin/python
from Engine import *
from HelperEngine import HelperEngine
import loader
import board3d as go_board

class KGSEngine(BaseEngine):
    def __init__(self, engine):
        self.engine = engine
        self.helper = HelperEngine()

    # subclasses must override this
    def name(self):
        return self.engine.name()

    # subclasses must override this
    def version(self):
        return self.engine.version()

    def set_board_size(self, N):
        return self.engine.set_board_size(N) and \
               self.helper.set_board_size(N)

    def clear_board(self):
        self.engine.clear_board()
        self.helper.clear_board()

    def set_komi(self, komi):
        self.engine.set_komi(komi)
        self.helper.set_komi(komi)

    def player_passed(self, color):
        self.engine.player_passed(color)
        self.helper.player_passed(color)

    def stone_played(self, x, y, color):
        self.engine.stone_played(x, y, color)
        self.helper.stone_played(x, y, color)

    def generate_move(self, color, cleanup=False):
        pass_checking_level = 5
        cleanup_level = 10
        self.helper.set_level(cleanup_level if cleanup else pass_checking_level)

        move = self.helper.generate_move(color, cleanup)
        if move.is_resign():
            return Move.Resign
        elif move.is_pass() or cleanup:
            self.engine.move_was_played(move, color)
            return move
        else:
            self.helper.undo()

        move = self.engine.generate_move(color)
        if move.is_play():
            self.helper.stone_played(move.x, move.y, color)
        elif move.is_pass():
            self.helper.player_passed(color)

        return move

    def undo(self):
        self.engine.undo()
        self.helper.undo()

    def quit(self):
        self.engine.quit()
        self.helper.quit()

    def supports_final_status_list(self):
        return True

    def final_status_list(self, status):
        print("Final status list is done by GNU Go")
        return self.helper.final_status_list(status)

    def final_score(self):
        return self.helper.final_score()

    def get_last_move_probs(self):
        return self.engine.get_last_move_probs()

    def toggle_kibitz_mode(self):
        return self.engine.toggle_kibitz_mode()

    def get_board_vis(self):
        return self.engine.get_board_vis()

if __name__ == '__main__':
    import GTP
    fclient = GTP.redirect_all_output("log_engine.txt")

    from GTP import GTP
    from TFEngine import TFEngine
    import loader

    engine = KGSEngine(TFEngine("EJEngine", loader.load_model_from_file(gvg.nn_type)))

    gtp = GTP(engine, fclient)
    gtp.loop()
