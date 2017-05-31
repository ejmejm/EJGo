import tkinter as tk
import play_game as controller
import numpy as np

#Most of code used from http://stackoverflow.com/questions/4954395/create-board-game-like-grid-in-python

class GameBoard(tk.Frame):
    def __init__(self, parent, rows=18, columns=18, size=40, color1="tan", color2="tan"):

        self.offset = 4
        self.stone_size = size - self.offset
        self.rows = rows
        self.columns = columns
        self.size = size
        self.color1 = color1
        self.color2 = color2
        self.pieces = {}

        canvas_width = columns * size
        canvas_height = rows * size

        tk.Frame.__init__(self, parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=1,
                                width=canvas_width, height=canvas_height, background="tan")
        self.canvas.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # this binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind("<Configure>", self.refresh)

        def click(event):
            print(event.x, self.offset, self.size)
            x = int((event.x + self.size/2) / self.size)
            y = self.columns - int((event.y - self.size*1.5) / self.size)
            print(x, ", ", y)
            advance_turn(x, y)

        self.canvas.bind("<Button 1>", click)

    def addpiece(self, name, image, row=0, column=0):
        #Add a piece to the playing board
		#TODO: Add image resizing
        self.canvas.create_image(0,0, image=image, tags=(name, "piece"), anchor="c")
        self.placepiece(name, row, column)

    def placepiece(self, name, row, column):
        #Place a piece at the given row/column
        self.pieces[name] = (row, column)
        x0 = (column * self.size) + self.offset * 10
        y0 = (row * self.size) + self.offset * 10
        self.canvas.coords(name, x0, y0)

    def refresh(self, event):
        print(event)
        #Redraw the board, possibly in response to window being resized
        xsize = int((event.width-1) / self.columns) - self.offset
        ysize = int((event.height-1) / self.rows) - self.offset
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x1 = (col * self.size) + self.offset * 10
                y1 = (row * self.size) + self.offset * 10
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=2, fill=color, tags="square")
                color = self.color1 if color == self.color2 else self.color2
        for name in self.pieces:
            self.placepiece(name, self.pieces[name][0], self.pieces[name][1])
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

def advance_turn(x, y):
    controller.take_turn(np.array([x, y]), set_stones)

def place_stone(x, y, player):
    if player == -1:
        ui_board.addpiece(str(x) + " " + str(y), player1, ui_board.columns - y, x)
    elif player == 1:
        ui_board.addpiece(str(x) + " " + str(y), player2, ui_board.columns - y, x)

def set_stones(board):
    for x in range(len(board)):
        for y in range(len(board[x])):
            ui_board.canvas.delete(str(x) + " " + str(y))
            place_stone(x, y, board[x][y])

root = tk.Tk()
ui_board = GameBoard(root)
ui_board.pack(side="top", fill="both", expand="true", padx=4, pady=4)
player1 = tk.PhotoImage(file="imgs/black_stone.png")
player2 = tk.PhotoImage(file="imgs/white_stone.png")
root.mainloop()
