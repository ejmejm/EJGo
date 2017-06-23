import numpy as np
import queue
import global_vars_go as gvg
from sgfmill.sgfmill import sgf_moves

empty = gvg.empty
filled = gvg.filled
color_to_play = gvg.kgs_black # Used to track stone color for KGS Engine

def make_move(board, move, player, enemy, debug=False):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)
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
    if group_captures == 0 and check_liberties(board, move) == 0:
        board[move[0]][move[1]][player] = empty
        if debug == True:
            print("ERROR! Illegal suicide move")
        return None

    # if color_to_play == gvg.kgs_black:
    #     color_to_play = gvg.kgs_white
    # else:
    #     color_to_play = gvg.kgs_black

    return board

def check_liberties(board, position):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)
    if board[position[0]][position[1]][gvg.bot_channel] == 1:
        player = gvg.bot_channel
        enemy = gvg.player_channel
    elif board[position[0]][position[1]][gvg.player_channel] == 1:
        player = gvg.player_channel
        enemy = gvg.bot_channel
    else:
        print("ERROR! Cannot check the liberties of an empty space")
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

def remove_stones(board, position):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)
    if board[position[0]][position[1]][gvg.bot_channel] == filled:
        player = gvg.bot_channel
        enemy = gvg.player_channel
    elif board[position[0]][position[1]][gvg.player_channel] == filled:
        player = gvg.player_channel
        enemy = gvg.bot_channel
    else:
        print("ERROR! Cannot check the liberties of an empty space")
        return;

    board_check = np.empty((gvg.board_size, gvg.board_size))
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
        board[c_move[0]][c_move[1]][player] = empty
    return board

def switch_player_perspec(board):
    board = board.reshape(gvg.board_size, gvg.board_size, gvg.board_channels)
    for i in range(len(board)):
        for j in range(len(board[i])):
            tmp = board[i][j][gvg.player_channel]
            board[i][j][gvg.player_channel] = board[i][j][gvg.bot_channel]
            board[i][j][gvg.bot_channel] = tmp
    return board

def setup_board(game):
    #color_to_play = gvg.kgs_black
    #Switch which color is which channel when the channels are switched
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

    return board.astype(int)

def empty_board(color): # Color is the bot's color
    color_to_play = color
    return np.zeros((gvg.board_size, gvg.board_size, gvg.board_channels))

def set_color(color):
    if color_to_play is None:
        color_to_play = color

def show_board(board):
    for i in range(board.shape[0]):
        print()
        for j in range(board.shape[1]):
            if(board[j, gvg.board_size-1-i, gvg.bot_channel] == gvg.filled):
                print("X ", end='')
            elif(board[j, gvg.board_size-1-i, gvg.player_channel] == gvg.filled):
                print("O ", end='')
            else:
                print(". ", end='')
