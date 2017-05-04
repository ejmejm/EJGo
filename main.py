from sgfmill.sgfmill import sgf


with open("testsgf.sgf", "rb") as f:
    game = sgf.Sgf_game.from_bytes(f.read())

winner = game.get_winner()
board_size = game.get_size()
root_node = game.get_root()
b_player = root_node.get("PB")
w_player = root_node.get("PW")
for node in game.get_main_sequence():
    if node.get_move()[1] != None:
        x = node.get_move()[1]
        print(x[0], ", ", x[1])



print("Test")
