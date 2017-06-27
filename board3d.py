import numpy as np
import queue
import time
import global_vars_go as gvg
from sgfmill.sgfmill import sgf_moves

empty = gvg.empty
filled = gvg.filled
color_to_play = gvg.kgs_black # Used to track stone color for KGS Engine

prev_moves = [[-10, -10], [-10, -10], [-10, -10], [-10, -10]]

prev_boards = [np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels)), \
              np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels))] # Keep track of previous board to avoid Ko violation

last_liberties = np.zeros((gvg.board_size, gvg.board_size, 4))

prev_liberties = [np.zeros((gvg.board_size, gvg.board_size, 4)), \
              np.zeros((gvg.board_size, gvg.board_size, 4))]

def reset_liberties():
    last_liberties = np.zeros((gvg.board_size, gvg.board_size, 4))
    prev_liberties = [np.zeros((gvg.board_size, gvg.board_size, 4)), \
                  np.zeros((gvg.board_size, gvg.board_size, 4))]

def make_move(board, move, player, enemy, debug=False):
    global last_liberties
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)
    backup_board = np.copy(prev_boards[1])
    backup_liberty = np.copy(prev_liberties[1])
    prev_boards[1] = np.copy(prev_boards[0]) # Update previous board to the current board
    prev_boards[0] = np.copy(board)
    prev_liberties[1] = np.copy(prev_liberties[0]) # Update previous liberties to the current liberties
    prev_liberties[0] = np.copy(last_liberties)

    board[move[0]][move[1]][player] = filled
    marked_to_check = []

    group_captures = 0
    if move[0] + 1 <= 18:
        if board[move[0]+1][move[1]][enemy] and check_liberties(board, np.array([move[0]+1, move[1]])) == 0:
                remove_stones(board, np.array([move[0]+1, move[1]]))
                group_captures += 1
        else:
            marked_to_check.append((move[0]+1, move[1]))
    if move[0] - 1 >= 0:
        if board[move[0]-1][move[1]][enemy] == filled and check_liberties(board, np.array([move[0]-1, move[1]])) == 0:
            remove_stones(board, np.array([move[0]-1, move[1]]))
            group_captures += 1
        else:
            marked_to_check.append((move[0]-1, move[1]))
    if move[1] + 1 <= 18:
        if board[move[0]][move[1]+1][enemy] == filled and check_liberties(board, np.array([move[0], move[1]+1])) == 0:
            remove_stones(board, np.array([move[0], move[1]+1]))
            group_captures += 1
        else:
            marked_to_check.append((move[0], move[1]+1))
    if move[1] - 1 >= 0:
        if board[move[0]][move[1]-1][enemy] == filled and check_liberties(board, np.array([move[0], move[1]-1])) == 0:
            remove_stones(board, np.array([move[0], move[1]-1]))
            group_captures += 1
        else:
            marked_to_check.append((move[0], move[1]-1))

    # Update liberties

    if legal_move(board, move, move_made=True, captures=group_captures) == False: # If the move made is illegal
        board = np.copy(prev_boards[0]) # Undo it
        prev_boards[0] = np.copy(prev_boards[1])
        prev_boards[1] = backup_board
        last_liberties = np.copy(prev_liverties[0])
        prev_liverties[0] = np.copy(prev_liverties[1])
        prev_liverties[1] = backup_liberty
        return None

    prev_moves[3] = prev_moves[2]
    prev_moves[2] = prev_moves[1]
    prev_moves[1] = prev_moves[0]
    prev_moves[0] = move

    return board

def legal_move(orig_board, move, move_made=False, player=None, captures=None, debug=False): # If it is legal to make a move in a space
    board = np.copy(orig_board)

    # If the player whose made the move has not been given, find who it was (assumes move has already been made)
    if player is None:
        if board[move[0]][move[1]][gvg.bot_channel] == gvg.filled:
            player = gvg.bot_channel
            enemy = gvg.player_channel
        elif board[move[0]][move[1]][gvg.player_channel] == gvg.filled:
            player = gvg.player_channel
            enemy = gvg.bot_channel
        else:
            if debug == True:
                print("ERROR! Cannot check for (il)legal move at empty space, (", move[0]+1, ", ", move[1]+1, ")")
            return 0
    else:
        if player == gvg.bot_channel:
            enemy = gvg.player_channel
        else:
            enemy = bot_channel

    # If a move has not been made, make it and get number of captures
    if move_made == False:
        board[move[0]][move[1]][player] = filled

        group_captures = 0
        if move[0] + 1 <= 18 and board[move[0]+1][move[1]][enemy] == filled and check_liberties(board, np.array([move[0]+1, move[1]])) == 0:
            remove_stones(board, np.array([move[0]+1, move[1]]))
            group_captures += 1
        if move[0] - 1 >= 0 and board[move[0]-1][move[1]][enemy] == filled and check_liberties(board, np.array([move[0]-1, move[1]])) == 0:
            remove_stones(board, np.array([move[0]-1, move[1]]))
            group_captures += 1
        if move[1] + 1 <= 18 and board[move[0]][move[1]+1][enemy] == filled and check_liberties(board, np.array([move[0], move[1]+1])) == 0:
            remove_stones(board, np.array([move[0], move[1]+1]))
            group_captures += 1
        if move[1] - 1 >= 0 and board[move[0]][move[1]-1][enemy] == filled and check_liberties(board, np.array([move[0], move[1]-1])) == 0:
            remove_stones(board, np.array([move[0], move[1]-1]))
            group_captures += 1

        captures = group_captures

    # If the move has been made, check how many captures
    if captures is None:
        captures = check_captures(board, move)

    if captures == 0 and check_liberties(board, move) == 0:
        if debug == True:
            print("ERROR! Illegal suicide move")
        return False
    elif (board == prev_boards[1]).all(): # If Ko
        if debug == True:
            print("ERROR! Illegal Ko violation")
        return False

    return True

def check_captures(orig_board, move, debug=False):
    board = np.copy(orig_board)

    if board[move[0]][move[1]][gvg.bot_channel] == gvg.filled:
        player = gvg.bot_channel
        enemy = gvg.player_channel
    elif board[move[0]][move[1]][gvg.player_channel] == gvg.filled:
        player = gvg.player_channel
        enemy = gvg.bot_channel
    else:
        if debug == True:
            print("ERROR! Cannot check the captures of an empty space at, (", move[0]+1, ", ", move[1]+1, ")")
        return 0

    captures = 0
    if move[0] + 1 <= 18 and board[move[0]+1][move[1]][enemy] == filled and check_liberties(board, np.array([move[0]+1, move[1]])) == 0:
        captures += remove_stones(board, np.array([move[0]+1, move[1]]))
    if move[0] - 1 >= 0 and board[move[0]-1][move[1]][enemy] == filled and check_liberties(board, np.array([move[0]-1, move[1]])) == 0:
        captures += remove_stones(board, np.array([move[0]-1, move[1]]))
    if move[1] + 1 <= 18 and board[move[0]][move[1]+1][enemy] == filled and check_liberties(board, np.array([move[0], move[1]+1])) == 0:
        captures += remove_stones(board, np.array([move[0], move[1]+1]))
    if move[1] - 1 >= 0 and board[move[0]][move[1]-1][enemy] == filled and check_liberties(board, np.array([move[0], move[1]-1])) == 0:
        captures += remove_stones(board, np.array([move[0], move[1]-1]))

    return captures

def set_liberty(move, val):
    last_liberties[move[0], move[1], 0] = gvg.empty
    last_liberties[move[0], move[1], 1] = gvg.empty
    last_liberties[move[0], move[1], 2] = gvg.empty
    last_liberties[move[0], move[1], 3] = gvg.empty
    val = min(val, 4)
    if val > 0:
        last_liberties[move[0], move[1], val - 1] = gvg.filled

def set_liberty_group(board, position, debug=False):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)

    if board[position[0]][position[1]][gvg.bot_channel] == gvg.filled:
        player = gvg.bot_channel
        enemy = gvg.player_channel
    elif board[position[0]][position[1]][gvg.player_channel] == gvg.filled:
        player = gvg.player_channel
        enemy = gvg.bot_channel
    else:
        set_liberty(move, 0)
        return;

    board_check = np.empty((gvg.board_size, gvg.board_size))
    board_check.fill(False)
    positions = queue.Queue()
    positions.put(position)
    board_check[position[0]][position[1]] = True

    liberties = 0
    while positions.empty() == False:
        c_move = positions.get()
        if c_move[0] + 1 <= 18 and board_check[c_move[0]+1][c_move[1]] == False:
            if board[c_move[0]+1][c_move[1]][player] == filled:
                positions.put(np.array([c_move[0]+1, c_move[1]]))
            elif board[c_move[0]+1][c_move[1]][enemy] == empty:
                liberties += 1
            board_check[c_move[0]+1][c_move[1]] = True
        if c_move[0] - 1 >= 0 and board_check[c_move[0]-1][c_move[1]] == False:
            if board[c_move[0]-1][c_move[1]][player] == filled:
                positions.put(np.array([c_move[0]-1, c_move[1]]))
            elif board[c_move[0]-1][c_move[1]][enemy] == empty:
                liberties += 1
            board_check[c_move[0]-1][c_move[1]] = True
        if c_move[1] + 1 <= 18 and board_check[c_move[0]][c_move[1]+1] == False:
            if board[c_move[0]][c_move[1]+1][player] == filled:
                positions.put(np.array([c_move[0], c_move[1]+1]))
            elif board[c_move[0]][c_move[1]+1][enemy] == empty:
                liberties += 1
            board_check[c_move[0]][c_move[1]+1] = True
        if c_move[1] - 1 >= 0 and board_check[c_move[0]][c_move[1]-1] == False:
            if board[c_move[0]][c_move[1]-1][player] == filled:
                positions.put(np.array([c_move[0], c_move[1]-1]))
            elif board[c_move[0]][c_move[1]-1][enemy] == empty:
                liberties += 1
            board_check[c_move[0]][c_move[1]-1] = True
    return liberties

def check_liberties(board, position, debug=False):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)

    if board[position[0]][position[1]][gvg.bot_channel] == gvg.filled:
        player = gvg.bot_channel
        enemy = gvg.player_channel
    elif board[position[0]][position[1]][gvg.player_channel] == gvg.filled:
        player = gvg.player_channel
        enemy = gvg.bot_channel
    else:
        if debug:
            print("ERROR! Cannot check the liberties of an empty space at, (", position[0]+1, ", ", position[1]+1, ")")
        return;

    board_check = np.empty((gvg.board_size, gvg.board_size))
    board_check.fill(False)
    positions = queue.Queue()
    positions.put(position)
    board_check[position[0]][position[1]] = True

    liberties = 0
    while positions.empty() == False:
        c_move = positions.get()
        if c_move[0] + 1 <= 18 and board_check[c_move[0]+1][c_move[1]] == False:
            if board[c_move[0]+1][c_move[1]][player] == filled:
                positions.put(np.array([c_move[0]+1, c_move[1]]))
            elif board[c_move[0]+1][c_move[1]][enemy] == empty:
                liberties += 1
            board_check[c_move[0]+1][c_move[1]] = True
        if c_move[0] - 1 >= 0 and board_check[c_move[0]-1][c_move[1]] == False:
            if board[c_move[0]-1][c_move[1]][player] == filled:
                positions.put(np.array([c_move[0]-1, c_move[1]]))
            elif board[c_move[0]-1][c_move[1]][enemy] == empty:
                liberties += 1
            board_check[c_move[0]-1][c_move[1]] = True
        if c_move[1] + 1 <= 18 and board_check[c_move[0]][c_move[1]+1] == False:
            if board[c_move[0]][c_move[1]+1][player] == filled:
                positions.put(np.array([c_move[0], c_move[1]+1]))
            elif board[c_move[0]][c_move[1]+1][enemy] == empty:
                liberties += 1
            board_check[c_move[0]][c_move[1]+1] = True
        if c_move[1] - 1 >= 0 and board_check[c_move[0]][c_move[1]-1] == False:
            if board[c_move[0]][c_move[1]-1][player] == filled:
                positions.put(np.array([c_move[0], c_move[1]-1]))
            elif board[c_move[0]][c_move[1]-1][enemy] == empty:
                liberties += 1
            board_check[c_move[0]][c_move[1]-1] = True
    return liberties

def remove_stones(board, position, count_only=False):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)
    if board[position[0]][position[1]][gvg.bot_channel] == filled:
        player = gvg.bot_channel
        enemy = gvg.player_channel
    elif board[position[0]][position[1]][gvg.player_channel] == filled:
        player = gvg.player_channel
        enemy = gvg.bot_channel
    else:
        print("ERROR! Cannot remove stones at the empty spot, (", move[0]+1, ", ", move[1]+1, ")")
        return;

    board_check = np.empty((gvg.board_size, gvg.board_size))
    captures = 0
    board_check.fill(False)
    positions = queue.Queue()
    positions.put(position)
    board_check[position[0]][position[1]] = True

    while positions.empty() == False:
        c_move = positions.get()
        if c_move[0] + 1 <= 18 and board_check[c_move[0]+1][c_move[1]] == False:
            if board[c_move[0]+1][c_move[1]][player] == filled:
                positions.put(np.array([c_move[0]+1, c_move[1]]))
            board_check[c_move[0]+1][c_move[1]] = True
        if c_move[0] - 1 >= 0 and board_check[c_move[0]-1][c_move[1]] == False:
            if board[c_move[0]-1][c_move[1]][player] == filled:
                positions.put(np.array([c_move[0]-1, c_move[1]]))
            board_check[c_move[0]-1][c_move[1]] = True
        if c_move[1] + 1 <= 18 and board_check[c_move[0]][c_move[1]+1] == False:
            if board[c_move[0]][c_move[1]+1][player] == filled:
                positions.put(np.array([c_move[0], c_move[1]+1]))
            board_check[c_move[0]][c_move[1]+1] = True
        if c_move[1] - 1 >= 0 and board_check[c_move[0]][c_move[1]-1] == False:
            if board[c_move[0]][c_move[1]-1][player] == filled:
                positions.put(np.array([c_move[0], c_move[1]-1]))
            board_check[c_move[0]][c_move[1]-1] = True
        if count_only == False:
            board[c_move[0]][c_move[1]][player] = empty
        captures += 1
    return captures

def encode_liberty_channels(board):
    liberty_channels = np.zeros((8, gvg.board_size, gvg.board_size))
    for i in range(gvg.board_size):
        for j in range(gvg.board_size):
            if board[i, j, gvg.bot_channel] == filled:
                liberties = min(check_liberties(board, (i, j), True), 4)
                liberty_channels[liberties - 1, i, j] = gvg.filled
            elif board[i, j, gvg.player_channel] == filled:
                liberties = min(check_liberties(board, (i, j), True), 4)
                liberty_channels[liberties + 3, i, j] = gvg.filled
    return liberty_channels

def encode_capture_channels(board):
    capture_channels = np.zeros((8, gvg.board_size, gvg.board_size))
    for i in range(gvg.board_size):
        for j in range(gvg.board_size):
            if board[i, j, gvg.bot_channel] == empty and \
                board[i, j, gvg.player_channel] == empty:

                board[i, j, gvg.bot_channel] = filled
                captures = min(check_captures(board, (i, j)), 4)
                board[i, j, gvg.bot_channel] = empty
                if captures > 0:
                    capture_channels[captures-1, i, j] = filled

                board[i, j, gvg.player_channel] = filled
                captures = min(check_captures(board, (i, j)), 4)
                board[i, j, gvg.player_channel] = empty
                if captures > 0:
                  capture_channels[captures + 3, i, j] = filled

    return capture_channels

def encode_border_channel(board):
    return np.ones((gvg.board_size, gvg.board_size))

def encode_empty_channel(board):
    empty_channel = np.zeros((gvg.board_size, gvg.board_size))
    for i in range(gvg.board_size):
        for j in range(gvg.board_size):
            if board[i, j, gvg.bot_channel] == empty and \
                board[i, j, gvg.player_channel] == empty:
                empty_channel[i, j] = filled

    return empty_channel

def encode_prev_moves_channels(board):
    prev_moves_channels = np.zeros((4, gvg.board_size, gvg.board_size))

    for i in range(4):
        if prev_moves[i][0] >= 0 and prev_moves[i][1] >= 0:
            prev_moves_channels[i, prev_moves[0], prev_moves[1]] = filled

    return prev_moves_channels

# Takes ~20 milliseconds on average
def get_encoded_board(board):
    enc_board = np.zeros((gvg.board_size, gvg.board_size, 20))
    for i in range(enc_board.shape[0]):
        for j in range(enc_board.shape[1]):
            # Border channel
            enc_board[i, j, gvg.border_channel] = filled
            if board[i, j, gvg.bot_channel] == filled:
                enc_board[i, j, gvg.bot_channel] = filled

                # Liberties
                liberties = min(check_liberties(board, (i, j)), 4)
                if liberties > 0:
                    enc_board[i, j, gvg.bot_liberty_channels[0] + liberties - 1] = filled
            elif board[i, j, gvg.player_channel] == filled:
                enc_board[i, j, gvg.player_channel] = filled

                # Liberties
                liberties = min(check_liberties(board, (i, j)), 4)
                if liberties > 0:
                    enc_board[i, j, gvg.player_liberty_channels[0] + liberties - 1] = filled
            else:
                enc_board[i, j, gvg.empty_channel] = filled

    for i in range(enc_board.shape[0]):
        for j in range(enc_board.shape[1]):
            if enc_board[i, j, gvg.empty_channel] == filled:
                board[i, j, gvg.bot_channel] = filled
                enemy = gvg.player_channel

                # Captures channel
                captures = 0

                if i+1 <= 18 and board[i+1, j, enemy] == filled and \
                enc_board[i+1, j, gvg.player_liberty_channels[1]] == 0 and \
                enc_board[i+1, j, gvg.player_liberty_channels[2]] == 0 and \
                enc_board[i+1, j, gvg.player_liberty_channels[3]] == 0:
                    captures = min(captures + remove_stones(board, np.array([i+1, j]), count_only=True), 4)
                if captures < 4 and i-1 >= 0 and board[i-1, j, enemy] == filled and \
                enc_board[i-1, j, gvg.player_liberty_channels[1]] == 0 and \
                enc_board[i-1, j, gvg.player_liberty_channels[2]] == 0 and \
                enc_board[i-1, j, gvg.player_liberty_channels[3]] == 0:
                    captures = min(captures + remove_stones(board, np.array([i-1, j]), count_only=True), 4)
                if captures < 4 and j+1 <= 18 and board[i, j+1, enemy] == filled and \
                enc_board[i, j+1, gvg.player_liberty_channels[1]] == 0 and \
                enc_board[i, j+1, gvg.player_liberty_channels[2]] == 0 and \
                enc_board[i, j+1, gvg.player_liberty_channels[3]] == 0:
                    captures = min(captures + remove_stones(board, np.array([i, j+1]), count_only=True), 4)
                if captures < 4 and j-1 >= 0 and board[i, j-1, enemy] == filled and \
                enc_board[i, j-1, gvg.player_liberty_channels[1]] == 0 and \
                enc_board[i, j-1, gvg.player_liberty_channels[2]] == 0 and \
                enc_board[i, j-1, gvg.player_liberty_channels[3]] == 0:
                    captures = min(captures + remove_stones(board, np.array([i, j-1]), count_only=True), 4)

                board[i, j, gvg.bot_channel] = empty

                if captures > 0:
                    enc_board[i, j, gvg.capture_channels[0] + captures - 1] = filled

    # Prev moves
    for i in range(4):
        if prev_moves[i][0] >= 0 and prev_moves[i][1] >= 0:
            enc_board[prev_moves[i][0], prev_moves[i][1], gvg.prev_moves_channels[i]] = filled

    return enc_board

def switch_player_perspec(board):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)
    for i in range(len(board)):
        for j in range(len(board[i])):
            tmp = board[i][j][gvg.player_channel]
            board[i][j][gvg.player_channel] = board[i][j][gvg.bot_channel]
            board[i][j][gvg.bot_channel] = tmp
    tmp = gvg.black_channel
    gvg.black_channel = gvg.white_channel
    gvg.white_channel = tmp

    return board

def setup_board(game):
    #color_to_play = gvg.kgs_black
    #Switch which color is which channel when the channels are switched
    reset_liberties()
    bc = gvg.black_channel
    gvg.black_channel = gvg.white_channel
    gvg.white_channel = bc
    preboard, plays = sgf_moves.get_setup_and_moves(game)
    rpreboard = preboard.board
    board = np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels))
    if len(plays) < 1: # Return an empty board if the game has no moves
        return board
    if plays[0][0] == "b":
        color_stone = gvg.bot_channel
    else:
        color_stone = gvg.player_channel
    for i in range(len(rpreboard)):
        for j in range(len(rpreboard[i])):
            if rpreboard[i][j] == "b":
                board[i][j][color_stone] = gvg.filled
                last_liberties[i, j, 3] = gvg.filled

    return board.astype(int)

def empty_board(color): # Color is the bot's color
    color_to_play = color
    reset_liberties()
    prev_boards = [np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels)), \
                  np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels))]
    return np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels))

def set_color(color):
    if color_to_play is None:
        color_to_play = color

# Prints ASCII representation of the board
def show_board(board):
    for i in range(board.shape[0]):
        print()
        for j in range(board.shape[1]):
            if(board[j, gvg.board_size-1-i, gvg.black_channel] == gvg.filled):
                print("X ", end='')
            elif(board[j, gvg.board_size-1-i, gvg.white_channel] == gvg.filled):
                print("O ", end='')
            else:
                print(". ", end='')

# Returns string representation of the board
def board_to_str(board):
    vis = ""
    for i in range(board.shape[0]):
        vis += "\n"
        for j in range(board.shape[1]):
            if(board[j, gvg.board_size-1-i, gvg.black_channel] == gvg.filled):
                vis += "X "
            elif(board[j, gvg.board_size-1-i, gvg.white_channel] == gvg.filled):
                vis += "O "
            else:
                vis += ". "
    return vis
