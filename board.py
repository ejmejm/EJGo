import numpy as np
import queue

def make_move(board, move, player):
    board[move[0]][move[1]] = player
    enemy = -player
    empty = 0

    group_captures = 0
    if move[0] + 1 <= 18 and board[move[0]+1][move[1]] == enemy and check_liberties(board, np.array([move[0]+1, move[1]])) == 0:
        remove_stones(board, np.array([move[0]+1, move[1]]))
        group_captures += 1
    if move[0] - 1 >= 0 and board[move[0]-1][move[1]] == enemy and check_liberties(board, np.array([move[0]-1, move[1]])) == 0:
        remove_stones(board, np.array([move[0]-1, move[1]]))
        group_captures += 1
    if move[1] + 1 <= 18 and board[move[0]][move[1]+1] == enemy and check_liberties(board, np.array([move[0], move[1]+1])) == 0:
        remove_stones(board, np.array([move[0], move[1]+1]))
        group_captures += 1
    if move[1] - 1 >= 0 and board[move[0]][move[1]-1] == enemy and check_liberties(board, np.array([move[0], move[1]-1])) == 0:
        remove_stones(board, np.array([move[0], move[1]-1]))
        group_captures += 1
    if group_captures == 0 and check_liberties(board, move) == 0:
        board[move[0]][move[1]] = empty
        print("ERROR! Illegal suicide move")
        return None

    return board

def check_liberties(board, position):
    player = board[position[0]][position[1]]
    if player == 0:
        print("ERROR! Cannot check the liberties of an empty space")
        return;
    empty = 0
    board_check = np.empty_like(board)
    board_check.fill(False)
    positions = queue.Queue()
    positions.put(position)
    board_check[position[0]][position[1]] = True

    liberties = 0
    while positions.empty() == False:
        c_move = positions.get()
        if c_move[0] + 1 <= 18 and board_check[c_move[0]+1][c_move[1]] == False:
            if board[c_move[0]+1][c_move[1]] == player:
                positions.put(np.array([c_move[0]+1, c_move[1]]))
            elif board[c_move[0]+1][c_move[1]] == empty:
                liberties += 1
            board_check[c_move[0]+1][c_move[1]] = True
        if c_move[0] - 1 >= 0 and board_check[c_move[0]-1][c_move[1]] == False:
            if board[c_move[0]-1][c_move[1]] == player:
                positions.put(np.array([c_move[0]-1, c_move[1]]))
            elif board[c_move[0]-1][c_move[1]] == empty:
                liberties += 1
            board_check[c_move[0]-1][c_move[1]] = True
        if c_move[1] + 1 <= 18 and board_check[c_move[0]][c_move[1]+1] == False:
            if board[c_move[0]][c_move[1]+1] == player:
                positions.put(np.array([c_move[0], c_move[1]+1]))
            elif board[c_move[0]][c_move[1]+1] == empty:
                liberties += 1
            board_check[c_move[0]][c_move[1]+1] = True
        if c_move[1] - 1 >= 0 and board_check[c_move[0]][c_move[1]-1] == False:
            if board[c_move[0]][c_move[1]-1] == player:
                positions.put(np.array([c_move[0], c_move[1]-1]))
            elif board[c_move[0]][c_move[1]-1] == empty:
                liberties += 1
            board_check[c_move[0]][c_move[1]-1] = True
    return liberties

def remove_stones(board, position):
    player = board[position[0]][position[1]]
    if player == 0:
        print("ERROR! Cannot remove an empty space")
        return;
    empty = 0
    board_check = np.empty_like(board)
    board_check.fill(False)
    positions = queue.Queue()
    positions.put(position)
    board_check[position[0]][position[1]] = True

    while positions.empty() == False:
        c_move = positions.get()
        if c_move[0] + 1 <= 18 and board_check[c_move[0]+1][c_move[1]] == False:
            if board[c_move[0]+1][c_move[1]] == player:
                positions.put(np.array([c_move[0]+1, c_move[1]]))
            board_check[c_move[0]+1][c_move[1]] = True
        if c_move[0] - 1 >= 0 and board_check[c_move[0]-1][c_move[1]] == False:
            if board[c_move[0]-1][c_move[1]] == player:
                positions.put(np.array([c_move[0]-1, c_move[1]]))
            board_check[c_move[0]-1][c_move[1]] = True
        if c_move[1] + 1 <= 18 and board_check[c_move[0]][c_move[1]+1] == False:
            if board[c_move[0]][c_move[1]+1] == player:
                positions.put(np.array([c_move[0], c_move[1]+1]))
            board_check[c_move[0]][c_move[1]+1] = True
        if c_move[1] - 1 >= 0 and board_check[c_move[0]][c_move[1]-1] == False:
            if board[c_move[0]][c_move[1]-1] == player:
                positions.put(np.array([c_move[0], c_move[1]-1]))
            board_check[c_move[0]][c_move[1]-1] = True
        board[c_move[0]][c_move[1]] = empty
    return board
