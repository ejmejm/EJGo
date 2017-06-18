import board3d as go_board
import numpy as np
import global_vars_go as gvg

def games_to_states(game_data):
    train_boards = []
    train_next_moves = []
    for game_index in range(len(game_data)):
        board = go_board.setup_board(game_data[game_index])
        for node in game_data[game_index].get_main_sequence():
            board = go_board.switch_player_perspec(board) # Changes player perspective, black becomes white and vice versa

            node_move = node.get_move()[1]
            if node_move is not None:
                train_boards.append(np.copy(board))
                next_move = np.zeros(gvg.board_size * gvg.board_size).reshape(gvg.board_size, gvg.board_size)
                next_move[node_move[0], node_move[1]] = gvg.filled # y = an array in the form [board_x_position, board_y_position]
                train_next_moves.append(next_move.reshape(gvg.board_size * gvg.board_size))

                board = go_board.make_move(board, node_move, gvg.bot_channel, gvg.player_channel) # Update board with new move
                if board is None:
                    print("ERROR! Illegal move, {}, while training".format(node_move))

    return train_boards, train_next_moves
