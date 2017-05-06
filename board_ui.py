import tkinter as tk

#Most of code used from http://stackoverflow.com/questions/4954395/create-board-game-like-grid-in-python

class GameBoard(tk.Frame):
    def __init__(self, parent, rows=18, columns=18, size=55, color1="bisque", color2="bisque"):

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
                                width=canvas_width, height=canvas_height, background="bisque")
        self.canvas.pack(side="top", fill="both", expand=True, padx=10, pady=10)

        # this binding will cause a refresh if the user interactively
        # changes the window size
        self.canvas.bind("<Configure>", self.refresh)

        def click(event):
            x, y = event.x, event.y
            print('{}, {}'.format(x, y))

        self.canvas.bind("<Button 1>", click)

    def addpiece(self, name, image, row=0, column=0):
        #Add a piece to the playing board
        self.canvas.create_image(0,0, image=image, tags=(name, "piece"), anchor="c")
        self.placepiece(name, row, column)

    def placepiece(self, name, row, column):
        #Place a piece at the given row/column
        self.pieces[name] = (row, column)
        x0 = (column * self.size) + 40
        y0 = (row * self.size) + 40
        self.canvas.coords(name, x0, y0)

    def refresh(self, event):
        #Redraw the board, possibly in response to window being resized
        xsize = int((event.width-1) / self.columns) - 4
        ysize = int((event.height-1) / self.rows) - 4
        self.size = min(xsize, ysize)
        self.canvas.delete("square")
        color = self.color2
        for row in range(self.rows):
            color = self.color1 if color == self.color2 else self.color2
            for col in range(self.columns):
                x1 = (col * self.size) + 40
                y1 = (row * self.size) + 40
                x2 = x1 + self.size
                y2 = y1 + self.size
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=2, fill=color, tags="square")
                color = self.color1 if color == self.color2 else self.color2
        for name in self.pieces:
            self.placepiece(name, self.pieces[name][0], self.pieces[name][1])
        self.canvas.tag_raise("piece")
        self.canvas.tag_lower("square")

root = tk.Tk()
board = GameBoard(root)
board.pack(side="top", fill="both", expand="true", padx=4, pady=4)
player1 = tk.PhotoImage(file="imgs/black_stone.gif")
player2 = tk.PhotoImage(file="imgs/white_stone.gif")
board.addpiece("player1", player1, 0,0)
board.addpiece("player2", player2, 1,0)
board.addpiece("player3", player1, 3,3)
board.addpiece("player4", player2, 3,0)
board.addpiece("player5", player1, 4,0)
board.addpiece("player6", player2, 2,3)
root.mainloop()
