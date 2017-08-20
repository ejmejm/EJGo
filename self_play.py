import GTP
fclient = GTP.redirect_all_output("log_engine.txt")

from GTP import GTP
from KGSEngine import KGSEngine
from TFEngine import TFEngine
import loader
import global_vars_go as gvg

engine = KGSEngine(TFEngine("EJEngine", loader.load_model_from_file(gvg.nn_type)))

gtp = GTP(engine, fclient)

gtp.set_board_size("boardsize " + str(gvg.board_size))
gtp.clear_board()
gtp.set_komi("komi 6.5")

passes = 0

while passes < 2:
    passes = 0
    move = gtp.generate_move("genmove black", returnVal=True)
    if move == "pass":
        print("black pass")
        passes += 1
    elif move == "resign":
        print("black resign")
        break
    else:
        print("black", move)
    move = gtp.generate_move("genmove white", returnVal=True)
    if move == "pass":
        print("white pass")
        passes += 1
    elif move == "resign":
        print("white resign")
        break
    else:
        print("white", move)

gtp.send_final_score()

print("Game Over")
