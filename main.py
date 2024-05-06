import pygame as pg
import numpy as np
import tkinter as tk
import os
from ImageProcessor import *
from StringRenderer import *
from tkinter import filedialog
from tkinter import Checkbutton
from ParametersManager import *
def getImagePath():
    root = tk.Tk()
    root.withdraw() #use to hide tkinter window
    file = filedialog.askopenfile(parent=root, initialdir=os.getcwd(), title='Please select an image')
    if file is None:
        return ""
    return file.name

def askForParameters():
    pg.init()
    gui = ParameterGUI(np.array(pg.display.get_desktop_sizes()[0])/3)
    gui.loop()
    pg.quit()
    return
#askForParameters()

#isColored = (input("Is you image in black and white ? (y/n)") != 'y')
n = 500
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
imax = 1500
running = True
while running:
    i += 1
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False

    scr.fill((0, 0, 0))
    scr.blit(s, scrsize-scrsize[1]/5)

    index = process.process()
    render.add(lastindex, index)
    lastindex = index
    scr.blit(render.getSurface(), (0, 0))
    scr.blit(font.render("Diameter = " + str(2*r) + " meters", True, (255, 255, 255)),datapos )
    scr.blit(font.render("Number of strings = " + str(i) + "/" + str((i//imax +1)*imax), True, (255, 255, 255)), datapos+ datastep)
    scr.blit(font.render("Total length = "+str(int(process.gettotlength()/2))+" diameters", True, (255, 255, 255)), datapos + 2*datastep)
    scr.blit(font.render("Mean early length = " + str(int(process.getmeanlength()/2*100)/100) + " diameters", True, (255, 255, 255)), datapos + 3*datastep)
    scr.blit(font.render("Mean early value = " + str(int(process.getmeanvalue()*100)/100), True,(255, 255, 255)), datapos + 4 * datastep)

    pg.display.flip()
    clock.tick(60)

    if i % 10 == 0:
        process.dispImage(s, int(scrsize[1] / 5), (0, 0))
    if i % imax == 0:
        input("i = "+str(i)+", press enter to continue")
pg.quit()


