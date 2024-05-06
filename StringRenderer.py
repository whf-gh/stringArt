import pygame as pg
import numpy as np

class StringRenderer:
    def __init__(self, squareSize, pins=500):
        real_r_value = 0.5
        real_line_width = 0.0002
        blur_effect = 1

        self.linewidth = real_line_width / real_r_value * squareSize
        pglinewidth = int(self.linewidth)+1+blur_effect
        self.opacity = self.linewidth/pglinewidth*0.7
        self.linewidth = pglinewidth


        self.linecolor = np.array((0, 0, 0, 255), dtype=np.uint8)
        self.backgroundcolor = np.array((255, 255, 255), dtype=np.uint8)

        self.s1 = pg.surface.Surface((squareSize, squareSize), pg.SRCALPHA)
        self.s1.fill(self.backgroundcolor)
        self.s2 = pg.surface.Surface((squareSize, squareSize), pg.SRCALPHA)
        self.s2.set_alpha(int(255*self.opacity))

        self.n = pins

        print(self.linewidth)
        """buildingpoints"""
        t = np.linspace(0, 2 * np.pi, self.n, endpoint=False)-np.pi/2
        self.r = squareSize/2 - 10
        self.c = np.array((squareSize/2, squareSize/2))
        self.points = np.vstack((- self.r * np.sin(t) + self.c[0], self.r * np.cos(t) + self.c[1])).T
    def add(self, index1, index2):
        self.s2.fill((0, 0, 0, 0))
        pg.draw.line(self.s2,  self.linecolor, self.points[index1], self.points[index2], self.linewidth)
        self.s1.blit(self.s2, (0, 0))
    def getSurface(self):
        return self.s1
