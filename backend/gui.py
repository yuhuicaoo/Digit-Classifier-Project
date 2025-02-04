import tkinter as tk
from PIL import Image, ImageDraw

class DigitalCanvas:
    def __init__(self, root, width=280, height=280):
        self.width = width
        self.height = height
        
        self.canvas = tk.Canvas(root, width=self.width, height=self.height, bg="black")
        self.canvas.pack()

        self.image = Image.new("L", (self.width, self.height), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1,y1,x2,y2, outline="white", width=10)
        self.draw.line([x1,y1,x2,y2], fill="white", width=10)
    
    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0,0,self.width, self.height], fill="black")

