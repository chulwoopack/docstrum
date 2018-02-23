import cv2
import numpy
import math

import colors
import geometry as g
from box import Box
from dimension import Dimension
from scipy import spatial

def threshold(image, threshold=colors.greyscale.MID_GREY, method=cv2.THRESH_BINARY_INV):
    retval, dst = cv2.threshold(image, threshold, colors.greyscale.WHITE, method)
    return dst

class Character:

    def __init__(self, x, y):

        self.coordinate = [x, y]
        self.x = x
        self.y = y

        self.nearestNeighbours = []
        self.parentWord = None

    def assignParentWord(self, word):

        self.parentWord = word
        self.parentWord.registerChildCharacter(self)

        for neighbour in self.nearestNeighbours:
            if neighbour.parentWord == None:
                neighbour.assignParentWord(self.parentWord)

    def toArray(self):
        return self.coordinate

    def __len__(self):
        return len(self.coordinate)

    def __getitem__(self, key):
        return self.coordinate.__getitem__(key)

    def __setitem__(self, key, value):
        self.coordinate.__setitem__(key, value)

    def __delitem__(self, key):
        self.coordinate.__delitem__(key)

    def __iter__(self):
        return self.coordinate.__iter__()

    def __contains__(self, item):
        return self.coordinate.__contains__(item)

    def paint(self, image, color=colors.YELLOW):

        pointObj = g.Point(self.coordinate)
        image = pointObj.paint(image, color)
        return image

class CharacterSet:

    def __init__(self, sourceImage):

        self.characters = self.getCharacters(sourceImage)
        self.NNTree = spatial.KDTree([char.toArray() for char in self.characters])
        #self.angles = []
        #self.distances = []

    def getCharacters(self, sourceImage):

        characters = []

        image = sourceImage.copy()
        image = threshold(image)
        image = threshold(image, cv2.THRESH_OTSU, method=cv2.THRESH_BINARY)

        if False:
            self.display(image)

        for contour in self.getContours(image):
            try:
                box = Box(contour)

                moments = cv2.moments(contour)
                centroidX = int( moments['m10'] / moments['m00'] )
                centroidY = int( moments['m01'] / moments['m00'] )
                character = Character(centroidX, centroidY)
                
            except ZeroDivisionError:
                continue
                
            if box.area > 50:
                characters.append(character)

        return characters

    def getContours(self, sourceImage, threshold=-1):

        image = sourceImage.copy()
        blobs = []
        topLevelContours = []

        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in range(len(hierarchy[0])):

            if len(contours[i]) > 2:    # 1- and 2-point contours have a divide-by-zero error in calculating the center of mass.

                # bind each contour with its corresponding hierarchy context description.
                obj = {'contour': contours[i], 'context': hierarchy[0][i]}
                blobs.append(obj)

        for blob in blobs:
            parent = blob['context'][3]
            if parent <= threshold: # no parent, therefore a root
                topLevelContours.append(blob['contour'])

        return topLevelContours

    def getWords(self):

        words = []
        k = 5
        
        # find the average distance between nearest neighbours
        NNDistances = []
        for character in self.characters:
            result = self.NNTree.query(character.toArray(), k=k)  # we only want nearest neighbour, but the first result will be the point matching itself.
            nearestNeighbourDistance = result[0][1]
            NNDistances.append(nearestNeighbourDistance)
        avgNNDistance = sum(NNDistances)/len(NNDistances)

        maxDistance = avgNNDistance*2
        #maxDistance = avgNNDistance*20000
        for character in self.characters:
            queryResult = self.NNTree.query(character.coordinate, k=k)
            distances = queryResult[0]
            neighbours = queryResult[1]
            for i in range(1,k):
                if distances[i] < maxDistance:
                    neighbour = self.characters[neighbours[i]]
                    character.nearestNeighbours.append(neighbour)

        for character in self.characters:
            if character.parentWord == None:
                if len(character.nearestNeighbours) >= 0:
                    word = Word([character])
                    word.findTuples()
                    words.append(word)
                    #print "Tuples: ", word.angles
                    
                    
                
        return words

    def paint(self, image, color=colors.BLUE):

        for character in self.characters:
            image = character.paint(image, color)    # draw a dot at the word's center of mass.

        return image

class Word:

    def __init__(self, characters=[]):
        
        self.characters = set(characters)
        self.angles = []
        self.distances = []

        for character in characters:
            character.assignParentWord(self)
            
    def findTuples(self):
        # Get tuple info ... 2/21/2018
        for character in self.characters:
            for neighbour in character.nearestNeighbours:
                line = g.Line([character, neighbour])
                angle = line.calculateAngle(line.start, line.end)
                delta = line.start-line.end
                distance = math.sqrt(delta.x**2 + delta.y**2)
                #print("START: ",line.start, " END: ", line.end, " DIST: ", distance," ANGLE_degree: ", angle.degrees(), "ANGLE_canonical: ", angle.canonical)
                self.angles.append(angle.canonical)
                #self.angles.append(angle.degrees())
                self.distances.append(distance)
                            
    def registerChildCharacter(self, character):
        
        self.characters.add(character)

    def paint(self, image, color=colors.YELLOW):

        for character in self.characters:
            image = character.paint(image, color)
            
            for neighbour in character.nearestNeighbours:
                line = g.Line([character, neighbour])
                image = line.paint(image, color)

        return image
        

