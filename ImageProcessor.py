from PIL import Image
from PIL import ImageFilter as IF
import numpy as np
import pygame as pg
from skimage.draw import line_aa
from scipy.ndimage import sobel, laplace
def gray(im):#Turning a bitmap image array into a black and white RGB format image array
    im = (255 * im / im.max()).astype(int)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret.astype(np.uint8)
def gradNorm(im):
    return np.sqrt(sobel(im, axis=0)**2 + sobel(im, axis=1)**2)
def getGradNormVariance(im):
    gnim = gradNorm(im)
    mean = np.mean(gnim)
    var = np.sqrt(np.sum((gnim - mean)**2/np.prod(gnim.shape)))
    return var
class ImageProcessor:
    def __init__(self, path, isColored, sizeProcessing=np.array((500,500)), pins=500):
        """Loading Image from path and turning into an array"""
        self.initialImage = Image.open(path).convert('L')
        self.initialImageArray = 1-np.array(self.initialImage).astype(float)/np.max(self.initialImage)

        """Defining if gradiant has to be taken instead"""
        var = getGradNormVariance(self.initialImageArray)
        print(var)
        if var < 0.85:
            self.initialImageArray = gradNorm(self.initialImageArray)
            self.initialImageArray = ((self.initialImageArray - self.initialImageArray.min())/(self.initialImageArray.max() - self.initialImageArray.min()))


        """Exctraction on the centered working square"""
        self.squareSize = np.min(self.initialImageArray.shape[:2])
        offset1 = int(0.5 * (self.initialImageArray.shape[0] - self.squareSize))
        offset2 = int(0.5 * (self.initialImageArray.shape[1] - self.squareSize))
        self.square = self.initialImageArray[offset1:offset1+self.squareSize, offset2:offset2+self.squareSize]
        """Resizing of the square"""


        """Pins circle related data"""
        self.c = np.array((self.squareSize, self.squareSize), dtype=float)/2
        self.r = self.squareSize/2-10
        self.n = pins
        t = np.linspace(0, 2*np.pi, self.n, endpoint=False)
        self.points = np.vstack((self.r * np.sin(t) + self.c[0], self.r * np.cos(t) + self.c[1])).T

        """Strings statistics related data"""
        self.totlength = 0
        self.pattern = [0]
        self.meanlength = 0
        self.meansgraduation = 0.05
        self.meanvalue = 0

        """Line hiding related data"""
        self.minvalue = -1
        self.maxdecrease = 0.1
        self.decreasestrength = 0.9

        """Searching parameters"""
        self.fasttestrate = 1500
        self.fasttestaccuracy = 50
        self.retestaccuracy = 1500
        self.minsegmentlength = self.squareSize*0.1


    def process(self):
        lastindex = self.pattern[-1]
        newindex, length, value = self.getagoodline(lastindex, q=self.fasttestrate)
        self.hide(self.points[newindex], self.points[lastindex])
        self.pattern.append(newindex)
        self.totlength += np.linalg.norm(self.points[newindex]-self.points[lastindex])/self.r
        self.meanlength += (length/self.r - self.meanlength)*self.meansgraduation
        self.meanvalue += (value - self.meanvalue) * self.meansgraduation
        return newindex
    def dispImage(self, surface, size, position):
        #Adding image on surface
        surface.blit(pg.transform.smoothscale(pg.surfarray.make_surface(gray(np.abs( self.initialImageArray - self.minvalue).T)), (size, size)), position)

    def getagoodline(self, index, q=100):
        sample = ((1-2*np.random.random(q))*self.n/q*np.linspace(100, 3, q)).astype(int)
        bestvalue = self.minvalue*(2*self.r+1)
        bestindex = 0
        bestlength = 0
        k = int(index-self.n/2)%self.n
        for i, ke in enumerate(sample):
            k = (k+ke)%self.n
            T = 2/(i+1)**(0.9)

            pt1 = self.points[index]
            pt2 = self.points[k]
            length = np.linalg.norm(pt1-pt2)
            if length > self.minsegmentlength:
                value = self.integratesegment(pt1, pt2, q=self.fasttestaccuracy)
                if value >= bestvalue or np.random.random() < np.exp((value - bestvalue)/T):
                    #accuvalue = self.integratesegment(pt1, pt2, q=self.retestaccuracy)
                    #if bestvalue < accuvalue:
                    bestindex = k
                    #bestvalue = accuvalue
                    bestvalue = value
                    bestlength = length
        if bestindex == index:
            print("ERROR")
        return bestindex, bestlength, bestvalue

    def integratesegment(self, pt1, pt2, q=100, length=0):#montecarlo integration
        s = 0
        rdm = np.random.random((q, 3))
        brownian_radius = 3*np.random.random()
        index = np.vstack(((rdm[:, 0]*pt2[0] + (1-rdm[:, 0])*pt1[0] + brownian_radius*rdm[:, 1]).astype(int), (rdm[:, 0]*pt2[1] + (1-rdm[:, 0])*pt1[1] + brownian_radius*rdm[:, 2]).astype(int))) .T #point random barycenter + brownian move
        for t in index:
            s += self.square[t[0], t[1]]
        return s/q

    def hide(self, pt1, pt2):
        rr, cc, val = line_aa(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))
        self.square[rr, cc] = (self.square[rr, cc]-self.minvalue)*(1-self.maxdecrease*val**self.decreasestrength) + self.minvalue

    def gettotlength(self):
        return self.totlength
    def getmeanlength(self):
        return self.meanlength
    def getmeanvalue(self):
        return self.meanlength