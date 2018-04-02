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
        self.orientations = []
        self.dists = []
        
        # PREPROCESSING START - NOISE REMOVAL
        ## Blurring
        #greyscaleImage = cv2.medianBlur(greyscaleImage,3)
        ## Closing
        kernel = numpy.ones((5,5),numpy.uint8)
        
        ## Opening
        #kernel = numpy.ones((5,5),numpy.uint8)
        #greyscaleImage = cv2.morphologyEx(greyscaleImage, cv2.MORPH_CLOSE, kernel)
        #greyscaleImage = cv2.morphologyEx(greyscaleImage, cv2.MORPH_CLOSE, kernel)
        
        #greyscaleImage = cv2.morphologyEx(greyscaleImage, cv2.MORPH_OPEN, kernel)
        #greyscaleImage = cv2.morphologyEx(greyscaleImage, cv2.MORPH_CLOSE, kernel)
        #self.display(greyscaleImage)
        # PREPROCESSING STOP
        
        colorImage = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

        if showSteps: self.display(greyscaleImage, title="Original Image")
    
        #################################
        # VERTICAL LINE REMOVAL - START #
        #################################
        '''
        #blurredImage = cv2.GaussianBlur(greyscaleImage,(5,5),0)
        #if showSteps: self.display(blurredImage, title="Gaussian-based Blurred Image")
        blurredImage = cv2.bilateralFilter(greyscaleImage,9,95,95)
        if showSteps: self.display(blurredImage, title="Bilateral-filter-based Blurred Image")
        _, binaryImage = cv2.threshold(blurredImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #binaryImage = cv2.adaptiveThreshold(blurredImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        if showSteps: self.display(binaryImage, title="Otsu-based Binarized Image")
        binaryImage = cv2.bitwise_not(binaryImage)
        if showSteps: self.display(binaryImage, title="Inverted Image")
        
#        kernel_size = (3,3)
        verticalsize = binaryImage.shape[0] / 90;
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,verticalsize))
        
        verticalMask = cv2.erode(binaryImage, kernel, (-1, -1))
        if showSteps: self.display(verticalMask, title="MORP. Erosed Image")  
        verticalMask = cv2.dilate(verticalMask, kernel, (-1, -1))
        if showSteps: self.display(verticalMask, title="MORP. Dilated Image") 
        verticalMask = cv2.blur(verticalMask, (9,9))
        if showSteps: self.display(verticalMask, title="Smoothened Vertical-line Candidates") 
        # Recursive 
        verticalMask = cv2.dilate(verticalMask, kernel, (-1, -1))
        if showSteps: self.display(verticalMask, title="MORP. Dilated Image_#2") 
        verticalMask = cv2.blur(verticalMask, (9,9))
        if showSteps: self.display(verticalMask, title="Smoothened Vertical-line Candidates_#2") 
        verticalMask = cv2.dilate(verticalMask, kernel, (-1, -1))
        if showSteps: self.display(verticalMask, title="MORP. Dilated Image_#3") 
        verticalMask = cv2.blur(verticalMask, (9,9))
        if showSteps: self.display(verticalMask, title="Smoothened Vertical-line Candidates_#3")
        verticalMask = cv2.dilate(verticalMask, kernel, (-1, -1))
        if showSteps: self.display(verticalMask, title="MORP. Dilated Image_#4") 
        verticalMask = cv2.blur(verticalMask, (9,9))
        if showSteps: self.display(verticalMask, title="Smoothened Vertical-line Candidates_#4")
        verticalMask = cv2.dilate(verticalMask, kernel, (-1, -1))
        if showSteps: self.display(verticalMask, title="MORP. Dilated Image_#5") 
        verticalMask = cv2.blur(verticalMask, (9,9))
        if showSteps: self.display(verticalMask, title="Smoothened Vertical-line Candidates_#5") 
        #verticalMask = cv2.adaptiveThreshold(verticalMask,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        #_, verticalMask = cv2.threshold(verticalMask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, verticalMask = cv2.threshold(verticalMask, 1, 255, cv2.THRESH_BINARY)
        if showSteps: self.display(verticalMask, title="Thresholded Vertical-line Candidates") 
        
        verticalMask_mask = numpy.ones(binaryImage.shape[:2], dtype="uint8") * 255
        verticalMask_contours,verticalMask_hierarchy = cv2.findContours(verticalMask, 1, 2)
        for cnt in verticalMask_contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if h>binaryImage.shape[0]/3:
                cv2.drawContours(verticalMask_mask, [cnt], -1, 0, -1)
        if showSteps: self.display(cv2.bitwise_not(verticalMask_mask), title="Final Vertical-lines") 
         
        binaryImage = cv2.bitwise_and(binaryImage, verticalMask_mask)
        if showSteps: self.display(binaryImage, title="Fully Preprocessed Image") 
        '''
        ###############################
        # VERTICAL LINE REMOVAL - END #
        ###############################
        
        #_,binaryImage = cv2.threshold(greyscaleImage, cv2.THRESH_OTSU, colors.greyscale.WHITE, cv2.THRESH_BINARY)
        _, binaryImage = cv2.threshold(greyscaleImage,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binaryImage = cv2.bitwise_not(binaryImage)
        if showSteps: self.display(binaryImage, title="Otsu-based Binarized Image") 
          
        self.characters = text.CharacterSet(binaryImage)
        #self.display(binaryImage)
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
        
        textlineImage = self.find_textline(colorImage)
        self.display(textlineImage, title="Found textlines")
        
        self.image = colorImage

        stopwatch.lap("finished analysing page")
        stopwatch.endRun()
        
        #self.drawTextLine(self.words,colorImage)
        #self.paint(self.image)
        
        #self.display_textline(textlineImage)
        
        #self.display(self.paint_textline(self.image))
        print "Done."

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
        
        dist_peaks = []
        n_copy = n.copy()
        n_copy[::-1].sort() # sort in reverse way
        THRESHOLD_DIST_WIDTH = 15
        for i in xrange(num_bins):
            _max_idx = numpy.where(n == n_copy[i])    # Find peak
            if len(_max_idx[0])>1:                    # If ties,
                _max = _max_idx[0][int(len(_max_idx[0])/2)]  # get middle
            else:
                _max = _max_idx[0][0]
            dist_peaks.append(int(_max+numpy.min(dist)))
        print ("Distance peaks: %s" %dist_peaks)
        '''
        first_group_offset = -1
        second_group_offset = -1
        _min = _max = dist_peaks[0]
        for i in xrange(len(dist_peaks)):
            #print("Ele: %d" %dist_peaks[i])
            if first_group_offset>-1 and second_group_offset>-1:
                break
            if _min <= dist_peaks[i] <= _max:
                #print("...within [%d,%d]" %(_min,_max))
                continue
            elif abs(dist_peaks[i] -_min) <= THRESHOLD_DIST_WIDTH:
                if dist_peaks[i]<_min:
                    _min = dist_peaks[i]
                    #print("...new min %d" %_min)
                elif _max < dist_peaks[i]:
                    _max = dist_peaks[i]
                    #print("...new max %d" %_max)    
                continue
            elif abs(_max - dist_peaks[i]) <= THRESHOLD_DIST_WIDTH:
                if _max < dist_peaks[i]:
                    _max = dist_peaks[i]
                    #print("...new max %d" %_max)   
                elif dist_peaks[i]<_min:
                    _min = dist_peaks[i]
                    #print("...new min %d" %_min)
                continue
            else:
                if first_group_offset == -1:
                    first_group_offset = i
                    #print("...found first group!")
                    _min = dist_peaks[i]
                    _max = dist_peaks[i]
                else:
                    second_group_offset = i
        print ("first group: %s and avg: %d" %(dist_peaks[:first_group_offset],numpy.mean(dist_peaks[:first_group_offset])))
        print ("second group: %s and avg: %d" %(dist_peaks[first_group_offset:second_group_offset],numpy.mean(dist_peaks[first_group_offset:second_group_offset])))

        '''
    
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
            #self.nnDistHist(dist_hist,path)
    
    ''' paint '''
    ''' color words '''
    def paint(self, image):

        #print len(self.words)
        for word in self.words:
            image = word.paint(image, colors.RED)

        return image
    
    def find_textline(self,image):
        image = image.copy()
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
