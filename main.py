import pygame as pg
import numpy as np
import tkinter as tk
import os
from ImageProcessor import *
from StringRenderer import *
from tkinter import filedialog
from tkinter import Checkbutton
from ParametersManager import *
from ParametersSetup import *
def getImagePath():
    root = tk.Tk()
    root.withdraw() #use to hide tkinter window
    file = filedialog.askopenfile(parent=root, initialdir=os.getcwd(), title='Please select an image')
    if file is None:
        return ""
    return file.name

def askForParameters():
    pg.init()
    gui = ParameterSetup(np.array(pg.display.get_desktop_sizes()[0])/3)
    gui.loop()
    pg.quit()
    return gui.parameters
parameters = askForParameters()

#isColored = (input("Is you image in black and white ? (y/n)") != 'y')
n = parameters.get('Number of Pins', 500)
if n == 0:
    n = 500
r = parameters.get('Radius', 50) / 100
if r == 0:
    r = 0.5
pg.init()
clock = pg.time.Clock()
scrsize = np.array(pg.display.get_desktop_sizes()[0], dtype=float)*0.9

process = ImageProcessor(getImagePath(), isColored=False, pins=n)
render = StringRenderer(scrsize[1], pins=n)
lastindex = 0

scr = pg.display.set_mode(scrsize)
s = pg.surface.Surface(scrsize/5)

scr.fill((255, 0, 0))
s.fill((0, 0, 0))
process.dispImage(s, int(scrsize[1]/5), (0, 0))
pg.display.set_icon(s)
pg.display.set_caption("StringArtProcess")

fontsize = int(scrsize[1]*0.03)
font = pg.font.SysFont("Arial", fontsize)
datapos = np.array((scrsize[0]*0.6, scrsize[1]*0.05))
datastep = np.array((0, fontsize*1.1))
i = 0
imax = parameters.get('Max Iterations', 1500)
if imax == 0:
    imax = 1500
running = True
nail_indices = []  # Initialize nail_indices outside the loop
nail_coordinates = []  # Initialize nail_coordinates outside the loop

# Function to calculate nail coordinates
def calculate_nail_coordinates(index, r, n):
    angle = 2 * np.pi * index / n
    x = r * np.cos(angle)
    y = r * np.sin(angle)
    return (x, y)

while running:
    i += 1
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    scr.fill((0, 0, 0))
    scr.blit(s, scrsize-scrsize[1]/5)
    if i == 1:
        nail_indices = []
        nail_coordinates = []

    nail_indices.append(lastindex)
    nail_coordinates.append(calculate_nail_coordinates(lastindex, r, n))
    index = process.process()
    render.add(lastindex, index)
    lastindex = index
    scr.blit(render.getSurface(), (0, 0))
    scr.blit(font.render("Diameter = " + str(2*r) + " meters", True, (255, 255, 255)), datapos )
    scr.blit(font.render("Nails = " + str(n), True, (255, 255, 255)), datapos + datastep)
    scr.blit(font.render("Number of strings = " + str(i) + "/" + str((i//imax +1)*imax), True, (255, 255, 255)), datapos+ 2*datastep)
    scr.blit(font.render("Total length = "+str(int(process.gettotlength()/2))+" diameters", True, (255, 255, 255)), datapos + 3*datastep)
    scr.blit(font.render("Mean early length = " + str(int(process.getmeanlength()/2*100)/100) + " diameters", True, (255, 255, 255)), datapos + 4*datastep)
    scr.blit(font.render("Mean early value = " + str(int(process.getmeanvalue()*100)/100), True,(255, 255, 255)), datapos + 5 * datastep)

    pg.display.flip()
    clock.tick(60)

    if i % 10 == 0:
        process.dispImage(s, int(scrsize[1] / 5), (0, 0))
    if i % imax == 0:
        button_font = pg.font.SysFont("Arial", int(scrsize[1]*0.02))
        button_continue = pg.Rect(scrsize[0]*0.4, scrsize[1]*0.8, 200, 50)
        button_exit = pg.Rect(scrsize[0]*0.6, scrsize[1]*0.8, 200, 50)
        
        pg.draw.rect(scr, (0, 255, 0), button_continue)
        pg.draw.rect(scr, (255, 0, 0), button_exit)
        
        scr.blit(button_font.render("Continue", True, (0, 0, 0)), (scrsize[0]*0.4 + 50, scrsize[1]*0.8 + 10))
        scr.blit(button_font.render("Exit", True, (0, 0, 0)), (scrsize[0]*0.6 + 70, scrsize[1]*0.8 + 10))
        
        pg.display.flip()
        
        waiting_for_input = True
        while waiting_for_input:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    waiting_for_input = False
                elif event.type == pg.MOUSEBUTTONDOWN:
                    if button_continue.collidepoint(event.pos):
                        waiting_for_input = False
                    elif button_exit.collidepoint(event.pos):
                        running = False
                    waiting_for_input = False

# Print nail_indices and nail_coordinates when the program quits
print("Nail Indices:", nail_indices)
print("Nail Coordinates:", nail_coordinates)

pg.quit()
