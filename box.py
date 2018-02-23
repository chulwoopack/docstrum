import cv2
import numpy
import math

import geometry as g
import colors

class EmptyObject:

    def __init__(self):
        pass

def distance(start, end):

    rise = float(end[1]) - float(start[1])
    run = float(end[0]) - float(start[0])

    distance = math.sqrt(rise**2 + run**2)

    return distance

def midpoint(start, end):

    midX = float(start[0]+end[0]) / 2
    midY = float(start[1]+end[1]) / 2

    return (midX, midY)

def angle(start, end):

    rise = float(end[1]) - float(start[1])
    run = float(end[0]) - float(start[0])

    radians = math.atan2(rise, run)
    degrees = math.degrees(radians)

    return degrees

class Box:

    def __init__(self, points):

        self.rect = cv2.minAreaRect(points)    # rect = ((center_x,center_y),(width,height),angle)
        self.points = self.rectToPoints(self.rect)

        self.setImportantPoints(self.points)    # sets up properties such as self.top.left, self.center.right, etc

        self.width = distance(self.center.left, self.center.right)
        self.height = distance(self.top.left, self.bottom.left)
        self.area = self.width * self.height
        self.angle = angle(self.top.left, self.top.right)

        self.words = []                         # children words, with a center of mass inside the box.
        self.isLine = False                     # temporary flag, until I set up a proper Lines class.

    def rectToPoints(self, rect):

        points = cv2.cv.BoxPoints(rect)             # Find four vertices of rectangle from above rect
        points = numpy.int0(numpy.around(points))   # Round the values and make them integers
        return points

    def setImportantPoints(self, points):
        # figures out which point is top-left, etc, and assigns each to one of: self.top.left, self.top.right,
        # self.bottom.left, and self.bottom.right

        points  = sorted(points, key=lambda point: point[0]) # sort by x position.
        left = sorted(points[:2], key=lambda point: point[1])  # [top-left, bottom-left]
        right = sorted(points[2:], key=lambda point: point[1]) # [top-right, bottom-right]

        self.top = EmptyObject()
        self.top.left = left[0]
        self.top.right = right[0]

        self.bottom = EmptyObject()
        self.bottom.left = left[1]
        self.bottom.right = right[1]

        self.center = EmptyObject()
        self.center.left = midpoint(self.top.left, self.bottom.left)
        self.center.right = midpoint(self.top.right, self.bottom.right)
        self.center.center = midpoint(self.center.left, self.center.right)

    def isTouchingEdge(self, shape, closenessThreshold=200):

        isTouching = False
        shape = (shape[1], shape[0])    # switch width and height so that it matches the format of a Point()

        for point in self.points:
            if point[0] <= (0 + closenessThreshold):
                isTouching = True
            elif point[1] <= (0 + closenessThreshold):
                isTouching = True
            elif point[0] >= (shape[0] - closenessThreshold):
                isTouching = True
            elif point[1] >= (shape[1] - closenessThreshold):
                isTouching = True

        return isTouching

    def contains(self, word):

        boxContour = numpy.array(self.points)
        retval = cv2.pointPolygonTest(numpy.array([self.points]), word.center, False)

        if retval > 0:
            return True
        else:
            return False

    def paint(self, image, color, width=5):

        #image = g.Point(self.center.center).paint(image, colors.RED)
        cv2.polylines(image, [self.points], True, color, width, cv2.CV_AA)

        #image = g.Point(self.center.left).paint(image, colors.BLUE)
        #image = g.Point(self.center.right).paint(image, colors.GREEN)

        return image
