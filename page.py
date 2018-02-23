import cv2
import math
import numpy
import subprocess
import os

import colors
import geometry as g
from box import Box
import text
from dimension import Dimension
from stopwatch import Stopwatch
import numpy
import matplotlib.pyplot as plt
import ntpath

stopwatch = Stopwatch()

class Page:

    def __init__(self, path, showSteps=False, saveDocstrum=False):

        stopwatch.reset(path)

        self.showSteps = showSteps
        self.saveDocstrum = saveDocstrum
        greyscaleImage = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        # PREPROCESSING START - NOISE REMOVAL
        greyscaleImage = cv2.medianBlur(greyscaleImage,3)
        
        # PREPROCESSING STOP
        colorImage = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

        if showSteps:
            self.display(greyscaleImage)

        self.characters = text.CharacterSet(greyscaleImage)
        stopwatch.lap("got characters")
        self.words = self.characters.getWords()
        stopwatch.lap("got words & tuples")
        #print "Tuple 1: (",self.words[1].angles[1], ", ", self.words[1].distances[1], ")"
        self.buildDocstrum(path)
        stopwatch.lap("built Docstrum")
        #theta = self.words[1].angles
        #r = self.words[1].distances
        #ax = plt.subplot(111,polar=True)
        #ax.scatter(theta,r)
        #plt.show()
        
        
        self.image = colorImage

        stopwatch.lap("finished analysing page")
        stopwatch.endRun()
        
    def buildDocstrum(self, path):
        theta = []
        r = []
        sz = 1
        for word in self.words:
            for angle in word.angles:
                #theta.append(numpy.pi+angle) # The second quadrant
                #print "word.angle = <<", angle, ">>"
                theta.append(1/2*numpy.pi-angle) # The first quadrant
                theta.append(3/2*numpy.pi-angle) # The third quadrant
            for distance in word.distances:
                r.append(distance)
                r.append(distance)
        ax = plt.subplot(111,polar=True)
        #print "theta = [", theta, "]"
        #print "r = [", r, "]"
        ax.scatter(theta,r,sz)
        if self.saveDocstrum:
            plt.savefig(os.path.join(os.path.abspath("./docstrums"),ntpath.basename(path)))
        if self.showSteps:
            plt.show()
            
        
    def paint(self, image):

        print len(self.words)
        for word in self.words:
            image = word.paint(image, colors.RED)

        return image

    def save(self, path):

        image = self.image.copy()
        image = self.paint(image)
        cv2.imwrite(path, image)

    def display(self, image, boundingBox=(800,800), title='Image'):

        stopwatch.pause()

        if boundingBox:
            maxDimension = Dimension(boundingBox[0], boundingBox[1])
            displayDimension = Dimension(image.shape[1], image.shape[0])
            displayDimension.fitInside(maxDimension)
            image = cv2.resize(image, tuple(displayDimension))

        cv2.namedWindow(title, cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow(title, image)
        cv2.waitKey()

        stopwatch.unpause()

    def show(self, boundingBox=None, title="Image"):    #textImage

        #image = numpy.zeros(self.image.shape, numpy.uint8)
        image = self.image.copy()
        
        image = self.paint(image)

        self.display(image, boundingBox, title)

    def extractWords(self, sourceImage):

        image = sourceImage.copy()
        image = threshold(image)

        tempImageFile = os.path.join('src', 'tempImage.tiff')
        tempTextFile = os.path.join('src', 'tempText')

        mask = numpy.zeros(image.shape, numpy.uint8)
        singleWord = numpy.zeros(image.shape, numpy.uint8)
