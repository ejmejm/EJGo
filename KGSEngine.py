#Source: https://github.com/TheDuck314/go-NN

#!/usr/bin/python
from Engine import *
import loader

class KGSEngine(BaseEngine):
    def __init__(self, engine):
        self.engine = engine

    # subclasses must override this
    def name(self):
        return self.engine.name()

    # subclasses must override this
    def version(self):
        return self.engine.version()

    def set_board_size(self, N):
        return self.engine.set_board_size(N)

    def clear_board(self):
        self.engine.clear_board()

    def set_komi(self, komi):
        self.engine.set_komi(komi)

    def player_passed(self, color):
        self.engine.player_passed(color)

    def stone_played(self, x, y, color):
        self.engine.stone_played(x, y, color)

    def generate_move(self, color, cleanup=False):
        move = self.engine.generate_move(color)
        return move

    def undo(self):
        self.engine.undo()

    def quit(self):
        self.engine.quit()

    def supports_final_status_list(self):
        return False

    def final_status_list(self, status):
        print("TODO: final_status_list(self, status)")

    def get_last_move_probs(self):
        return self.engine.get_last_move_probs()

    def toggle_kibitz_mode(self):
        return self.engine.toggle_kibitz_mode()

if __name__ == '__main__':
    import GTP
    fclient = GTP.redirect_all_output("log_engine.txt")

    from GTP import GTP
    from TFEngine import TFEngine
    import loader

    engine = KGSEngine(TFEngine("EJEngine", loader.load_model_from_file(gvg.nn_type)))

    gtp = GTP(engine, fclient)
    gtp.loop()
