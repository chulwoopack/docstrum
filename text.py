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

    ''' paint '''
    ''' paint a dot on the centroid of a character '''
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

    ''' getCharacters '''
    ''' This function (1) binarize a source image (2) get contours (characters) (3) get its centroid  '''
    def getCharacters(self, sourceImage):

        characters = []

        image = sourceImage.copy()
        image = threshold(image)
        image = threshold(image, cv2.THRESH_OTSU, method=cv2.THRESH_BINARY)

        if False:
            cv2.imshow('binarized', image)
            cv2.waitKey()

        for contour in self.getContours(image):
            try:
                box = Box(contour)

                moments = cv2.moments(contour)
                centroidX = int( moments['m10'] / moments['m00'] )
                centroidY = int( moments['m01'] / moments['m00'] )
                character = Character(centroidX, centroidY)
                
            except ZeroDivisionError:
                continue
                
            #if box.area > 50:
            if box.area > 10:
            #if True:
                characters.append(character)

        print "Total ", len(characters), " characters are found."
        return characters

    ''' getContours         '''
    ''' Input: Binary Image '''
    ''' Output: BLOBs       '''
    def getContours(self, sourceImage, threshold=-1):
        image = sourceImage.copy()
        blobs = []
        topLevelContours = []

        # cv2.findContours : It stores the (x,y) coordinates of the boundary of a shape. Here, contours are the boundaries of a shape with same intensity.
        # CHAIN_APPROX_NONE : All the boundary points are stored.
        # CHAIN_APPROX_SIMPLE : It removes all redundant points and compresses the contour, thereby saving memory.
        # hierarchy = [Next, Previous, First_Child, Parent]
        # REFERENCE : https://docs.opencv.org/3.1.0/d4/d73/tutorial_py_contours_begin.html
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

    ''' transitiveClosure '''
    ''' Obtain nearest-neighbor groups on the same text lines with the use of a transitive closure on within-line nearest neighbor pairings '''
    '''    
    def transitiveClosure(self):

        self.characters = sorted(self.characters, key=lambda char: (char.y, char.x))
#        self.characters = sorted(self.characters, key=lambda char: char.x)
        for idx, character in enumerate(self.characters):
            print "[",idx,"] character's nn info..", "(",character.x,",",character.y,")"
            character.nearestNeighbours = sorted(character.nearestNeighbours, key=lambda nn_char: nn_char.x)
            # for each character's nn [checking purpose.. it can be removed later]
            for idx_nn, character_nn in enumerate(character.nearestNeighbours):
                print "**[", idx_nn, "] nn info.. ", "(",character_nn.x,",",character_nn.y,")"

        within_line_nn_groups = []
        within_line_nn_group = []
        
        start_flag = True
        end_flag = False
        
        self.characters = sorted(self.characters, key=lambda char: (char.y, char.x))
#        self.characters = sorted(self.characters, key=lambda char: char.x)
        # for each character
        for idx, character in enumerate(self.characters):
            if start_flag:
                within_line_nn_group.append(character)
            character.nearestNeighbours = sorted(character.nearestNeighbours, key=lambda nn_char: nn_char.x)
            if len(character.nearestNeighbours)>0:
                # start char of group
                if start_flag:
                    within_line_nn_group.append(character.nearestNeighbours[0])
                    start_flag = False
                # end char of group
                elif ((idx+1)==len(self.characters) or len(character.nearestNeighbours)<2 or  character.nearestNeighbours[1].x != self.characters[idx+1].x):
                    end_flag = True
                    within_line_nn_groups.append(within_line_nn_group)
                    within_line_nn_group = []
                    start_flag = True
                # mid char of group
                else:
                    within_line_nn_group.append(character.nearestNeighbours[1])
        print "Found group: ", within_line_nn_groups
    '''
        
    ''' getWords '''
    ''' Find nearest neighbors '''
    ''' Input: Characters '''
    ''' Output: k-nearest neighbors '''
    def getWords(self):

        words = []
        k = 3
       
        # find the average distance between nearest neighbours
        NNDistances = []
        for character in self.characters:
            result = self.NNTree.query(character.toArray(), k=k)  # we only want nearest neighbour, but the first result will be the point matching itself.
            nearestNeighbourDistance = result[0][1]
            NNDistances.append(nearestNeighbourDistance)
        avgNNDistance = sum(NNDistances)/len(NNDistances)

        maxDistance = avgNNDistance*3
        #maxDistance = avgNNDistance*20000
        for character in self.characters:
            #print ("Finding a a nn of ",character.x,character.y)
            queryResult = self.NNTree.query(character.coordinate, k=k)
            distances = queryResult[0]
            neighbours = queryResult[1]
            #print("...",character.coordinate,"'s nn ... ")
            for i in range(1,k):
                if distances[i] < maxDistance:
                #if True:
                    # check if it is a nn in horizontal way
                    #print("is...",self.characters[neighbours[i]],"..and...",abs(self.characters[neighbours[i]].y-character.y))
                    if(abs(self.characters[neighbours[i]].y-character.y) < avgNNDistance/2):
                        neighbour = self.characters[neighbours[i]]
                        character.nearestNeighbours.append(neighbour)
                        #print (i,"th nn!", "dist:", distances[i], " neighbor:(",neighbour.x,",",neighbour.y,")")
        
        #self.characters = sorted(self.characters, key=lambda character: (character.y, character.x))    
        self.characters = sorted(self.characters, key=lambda character: (character.x))    
        for character in self.characters:
            #print ("Deciding wordness of (",character.x,character.y,")")
            if character.parentWord == None:
                #print ("(",character.x,character.y,") is a parent!")
                if len(character.nearestNeighbours) >= 0:
                    #print ("(",character.x,character.y,") is a word!!!!")
                    word = Word([character])
                    word.findTuples()
                    words.append(word)
        '''
        print "Total ", len(words), " words are found."
        for idx, word in enumerate(words):
            print "[",idx,"] word:"
            for idx_char, character in enumerate(word.characters):
                print "**[", idx_char, "] char info.. ", "(",character.x,",",character.y,")"
        '''        
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

    ''' paint '''
    ''' Draw a line between characters '''
    def paint(self, image, color=colors.YELLOW):

        for character in self.characters:
            image = character.paint(image, color)
            
            for neighbour in character.nearestNeighbours:
                line = g.Line([character, neighbour])
                image = line.paint(image, color)

        return image
        
#class Line:
#    def __init__(self, words=[]):
#        self.words = set(words)
#    def update():
        
        

