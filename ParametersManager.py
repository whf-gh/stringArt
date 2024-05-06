import pygame as pg
import numpy as np

class StringPreview:
    def __init__(self, squaresize=500, n=500):
        self.squaresize = squaresize

        self.opacity = 1
        self.linecolor = (0, 0, 0)
        self.linewidth = 1

        self.n = n
        self.N = 40

        self.rollStrings()
        self.buildsurfaces()
        self.refresh()
    def buildsurfaces(self):
        self.s = pg.surface.Surface((self.squaresize, self.squaresize))
        self.s1 = pg.surface.Surface((self.squaresize, self.squaresize), pg.SRCALPHA)
        self.s2 = pg.surface.Surface((self.squaresize, self.squaresize), pg.SRCALPHA)
        self.s2.set_alpha(int(255 * self.opacity))
    def refresh(self):
        self.s1.fill((0, 0, 0))
        pg.draw.circle(self.s1, (255, 255, 255), (self.squaresize / 2, self.squaresize / 2), self.squaresize / 2)
        for i in range(self.N-1):
            self.s2.fill((0, 0, 0, 0))
            pg.draw.line(self.s2, self.linecolor, self.getpoint(self.indexes[i]), self.getpoint(self.indexes[i+1]), self.linewidth)
            self.s1.blit(self.s2, (0, 0))
        self.s.blit(self.s1, (0, 0))
        pg.display.set_icon(self.s)
    def getpoint(self, index):
        return self.squaresize * 0.5 * (1 + np.array((np.cos(2 * np.pi * int(index*self.n)/self.n), np.sin(2 * np.pi * int(index*self.n)/self.n))))
    def setopacity(self, opacity):
        self.opacity = opacity

    def setResol(self, resol):
        self.squaresize = resol
        self.buildsurfaces()
    def rollStrings(self):
        self.indexes = np.random.random(self.N)
    def blit(self, window, size, pos):
        window.blit(pg.transform.smoothscale(self.s, (size, size)), pos)

class widgetManager:
    def __init__(self, position=(0, 0), size=100):
        self.param = {"CircleRadius": {"name": "Circle radius", "postcase": "(cm)", "value": 50.0, "maxvalue": 200, "step": 0.5},
                      "StringWidth": {"name": "String witdh", "postcase": " (mm)", "value": 1.0, "maxvalue": 5, "step": 0.05}}
        self.paramNumber = len(self.param.keys())
        self.fontsize = 30
        self.font = pg.font.SysFont("Arial", self.fontsize)

        self.position = np.array(position)
        self.size = size*self.fontsize

        self.surface = pg.surface.Surface((self.size, self.fontsize *( self.paramNumber + 1)))
        self.surface.fill((0, 0, 0))

        self.updatesurface()
        self.beenUpdate = True
    def write(self, data, pos, isSelected=False):
        self.surface.blit(self.font.render(data["name"]+" = "+str(data["value"])+" "+data["postcase"], True, (255, 255, 255) if isSelected else (220, 220, 220)), pos)

    def update(self, stringpreview):
        if self.beenUpdate:

            self.beenUpdate = False
    def updatesurface(self):
        for i, key in enumerate(self.param.keys()):
            self.write(self.param.get(key), np.array((0,i*self.fontsize)))

    def tick(self):
        mousepos = pg.mouse.get_pos()
    def disp(self, window):
        window.blit(self.surface, self.position)
class ParameterGUI:
    def __init__(self, size):
        pg.init()
        self.size = np.array(size, dtype=int)
        self.window = pg.display.set_mode(self.size)
        pg.display.set_caption("Parameter Manager")
        self.stringpreview = StringPreview()
        self.widgetmanager = widgetManager((self.size[1], self.size[1]*0.05))
    def loop(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            self.window.fill((0, 0, 0))
            self.stringpreview.blit(self.window, self.size[1], (0, 0))
            self.widgetmanager.disp(self.window)
            self.widgetmanager.update(self.stringpreview)
            pg.display.flip()
        pg.quit()
