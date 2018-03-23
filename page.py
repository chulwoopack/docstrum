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

import itertools # For mostcommon
import operator  # For mostcommon

stopwatch = Stopwatch()

class Page:

    def __init__(self, path, showSteps=False, saveDocstrum=False):

        stopwatch.reset(path)

        self.showSteps = showSteps
        self.saveDocstrum = saveDocstrum
        self.lines = []
        greyscaleImage = cv2.imread(path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        self.display(greyscaleImage)
        self.orientations = []
        self.dists = []
        
        # PREPROCESSING START - NOISE REMOVAL
        ## Blurring
        #greyscaleImage = cv2.medianBlur(greyscaleImage,3)
        ## Closing
        kernel = numpy.ones((5,5),numpy.uint8)
        
        ## Opening
        #kernel = numpy.ones((5,5),numpy.uint8)
        greyscaleImage = cv2.morphologyEx(greyscaleImage, cv2.MORPH_CLOSE, kernel)
        greyscaleImage = cv2.morphologyEx(greyscaleImage, cv2.MORPH_OPEN, kernel)
        #greyscaleImage = cv2.morphologyEx(greyscaleImage, cv2.MORPH_CLOSE, kernel)
        #self.display(greyscaleImage)
        # PREPROCESSING STOP
        
        colorImage = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

        if showSteps:
            self.display(greyscaleImage)

        self.characters = text.CharacterSet(greyscaleImage)
        stopwatch.lap("got characters")
        # words = [word, word, ..., word]
        # words = [append, count, extend, index, insert, pop, remove, reverse, sort]
        # word  = [angles, characters, distances, findTuples, paint, registerChildCharacter]
        # word  = [char, char, ..., char]
        # char = [nearestNeighbors, parentWord, x, y]
        self.words = self.characters.getWords()
        stopwatch.lap("got words & tuples")
        
        print "Total ", len(self.words), " words are found."
        #for idx, word in enumerate(self.words):
        #    print "[",idx,"] word:"
        #    for idx_char, character in enumerate(word.characters):
        #        print "**[", idx_char, "] char info.. ", "(",character.x,",",character.y,")"
        
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
        
        #self.drawTextLine(self.words,colorImage)
        #self.paint(self.image)
        self.paint_textline(self.image)
        #self.display(self.paint_textline(self.image))


    def most_common(L):
        # get an iterable of (item, iterable) pairs
        SL = sorted((x, i) for i, x in enumerate(L))
        # print 'SL:', SL
        groups = itertools.groupby(SL, key=operator.itemgetter(0))
        # auxiliary function to get "quality" for an item
        def _auxfun(g):
            item, iterable = g
            count = 0
            min_index = len(L)
            for _, where in iterable:
              count += 1
              min_index = min(min_index, where)
            # print 'item %r, count %r, minind %r' % (item, count, min_index)
            return count, -min_index
        # pick the highest-count/earliest item
        return max(groups, key=_auxfun)[0]

    def nnAngleHist(self, theta, path):
        #print "theta from hist: ", theta
        num_bins = 180
        n, bins, patches = plt.hist(theta, num_bins, facecolor='blue', alpha=0.5)
        if self.saveDocstrum:
            plt.savefig(os.path.join(os.path.abspath("./docstrums"),"ds_nnAngle_" + ntpath.basename(path)))
        plt.show()
        
    def nnDistHist(self, dist, path):
        num_bins = int(numpy.max(dist)-numpy.min(dist)+1)
        #num_bins = 2*int(numpy.max(dist)+1)
        #print("num_bins: ",num_bins)
        #num_bins = 180
        n, bins, patches = plt.hist(dist, num_bins, facecolor='orange', alpha=0.5)
        if self.saveDocstrum:
            plt.savefig(os.path.join(os.path.abspath("./docstrums"),"ds_nnDist_" + ntpath.basename(path)))
        plt.show()
    
    
    def buildDocstrum(self, path):
        theta = []
        theta_hist = []
        dist_hist = []
        r = []
        sz = 1
        for word in self.words:
            for angle in word.angles:
                #theta.append(numpy.pi+angle) # The second quadrant
                #print "word.angle = <<", angle, ">>"
                theta.append(1/2*numpy.pi-angle) # -pi/2 < x < pi/2 (1 and 4 quadrant)
                theta.append(3/2*numpy.pi-angle) # pi/2 < x < -pi/2 (2 and 3 quadrant)
                theta_hist.append(math.degrees(1/2*numpy.pi-angle))
            for distance in word.distances:
                r.append(distance)
                r.append(distance)
                dist_hist.append(distance)
        ax = plt.subplot(111,polar=True)
        #print("The peak of text-line orientation: ",self.most_common(theta_hist))
        #print("shape of dist_hist: ",numpy.shape(dist_hist))
        #print("The peak of within-line distance: ",self.most_common(dist_hist))
        self.orientations = theta_hist
        self.dists = dist_hist
        ax.scatter(theta,r,sz)
        if self.saveDocstrum:
            plt.savefig(os.path.join(os.path.abspath("./docstrums"),"ds_" + ntpath.basename(path)))
        if self.showSteps:
            plt.show()
            self.nnAngleHist(theta_hist,path)
            self.nnDistHist(dist_hist,path)
    
    ''' paint '''
    ''' color words '''
    def paint(self, image):

        print len(self.words)
        for word in self.words:
            image = word.paint(image, colors.RED)

        return image
    
    def paint_textline(self, image):
        ratio = 4.0/8.0
        #ratio = 4.0/4.0
        for word in self.words:
            #dir(word)
            #word.angles
            points = []
            multiplier = 1
            for character in word.characters:
                #print "(",character.x,", ",character.y,")"
                #print "nn: ", character.nearestNeighbors
                points.append([character.x, character.y])
            points.sort(key=lambda x: x[0])
            #print("points:",points)
            w = max(points,key=lambda x: x[0])[0]-min(points,key=lambda x: x[0])[0]
            #print("w:",w)
            h = max(points,key=lambda x: x[1])[1]-min(points,key=lambda x: x[1])[1]
            #print(h)
            dx, dy, x0, y0 = cv2.fitLine(numpy.array(points), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
            #print("dx:",dx,", dy:",dy,", x0:",x0,", y0:",y0)
            #start = (int(x0 - dx*w*ratio), int(y0 - dy*w*ratio))
            start = (int(min(points,key=lambda x: x[0])[0]),int((dy/dx)*(min(points,key=lambda x: x[0])[0]-x0)+y0))
            #end = (int(x0 + dx*w*ratio), int(y0 + dy*w*ratio))
            end = (int(max(points,key=lambda x: x[0])[0]),int((dy/dx)*(max(points,key=lambda x: x[0])[0]-x0)+y0))
            #print(start,end)
            self.lines.append(g.Line([start,end]))
            cv2.line(image, start, end, (0,255,255),2)
        return image

    def save(self, path):

        image = self.image.copy()
        image = self.paint(image)
        #image = self.paint_textline(image)
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
