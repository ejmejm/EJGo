#Source: https://github.com/TheDuck314/go-NN/blob/master/engine/HelperEngine.py

import subprocess
from GTP import *

# Using gnugo to determine when to pass and to play cleanup moves

def color_to_str(color):
    if color == Color.Black:
        return "black"
    elif color == Color.White:
        return "white"
    else:
        print("HelperEngine: ERROR! Could not identify color with channel", color)
        return "N/A"

class HelperEngine:
    def __init__(self, level=10):
        command = ["gnugo", "--mode", "gtp", "--level", str(level), "--chinese-rules", "--positional-superko"]
        self.proc = subprocess.Popen(command, bufsize=1, stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True) # bufsize=1 is line buffered

    def send_command(self, command):
        print("HelperEngine: sending command \"{}\"".format(command))
        self.proc.stdin.write(command)
        self.proc.stdin.write('\n')

        response = ""
        import sys
        while True:
            line = self.proc.stdout.readline()
            if line.startswith('='):
                response += line[2:]
            elif line.startswith('?'):
                print("HelperEngine: error response! line is \"{}\"".format(line))
                response += line[2:]
            elif len(line.strip()) == 0:
                # blank line ends response
                break
            else:
                response += line
        response = response.strip()
        print("HelperEngine: got response \"{}\"".format(response))
        return response

    def set_board_size(self, N):
        self.send_command("boardsize {}".format(N))
        return True # could parse helper response

    def clear_board(self):
        self.send_command("clear_board")

    def set_komi(self, komi):
        self.send_command("komi {}".format(komi))

    def player_passed(self, color):
        self.send_command("play {} pass".format(color_to_str(color)))

    def stone_played(self, x, y, color):
        self.send_command("play {} {}".format(color_to_str(color), str_from_coords(x, y)))

    def set_level(self, level):
        self.send_command("level {}".format(level))

    def generate_move(self, color, cleanup=False):
        cmd = "kgs-genmove_cleanup" if cleanup else "genmove"
        response = self.send_command("{} {}".format(cmd, color_to_str(color)))
        if 'pass' in response.lower():
            return Move.Pass
        elif 'resign' in response.lower():
            return Move.Resign
        else:
            x, y= coords_from_str(response)
            return Move(x, y)

    def undo(self):
        self.send_command('undo')

    def quit(self):
        pass

    def final_status_list(self, status):
        return self.send_command("final_status_list {}".format(status))

    def final_score(self):
        return self.send_command("final_score")

if __name__ == '__main__':
    helper = HelperEngine()

    helper.set_board_size(19)
    helper.clear_board()
    helper.set_komi(6.5)
    helper.stone_played(5, 5, Color.Black)
    move = helper.generate_move(Color.White)
    print("move =", move)
    helper.undo()
    move = helper.pick_move(Color.White)
    print("move =", move)
