import cv2

import colors
import geometry as g
from box import Box
import text

class NaiveMargin:
    """ This is a simple approximation of the margin, used to get rid of marginal noise."""

    def __init__(self, candidateLines):

        self.candidateLines = candidateLines

        fullLines = [line for line in self.candidateLines if 1280 < line.box.width < 1330]
        left =  g.Line([ line.box.center.left  for line in fullLines ])
        right = g.Line([ line.box.center.right for line in fullLines ])

        # Make sure that 'start' means the same end for both geometric lines. This fixes a frustrating problem,
        # where in some pages most lines wouldn't be picked up.
        if left.start[1] < left.end[1]:
            left.start, left.end = left.end, left.start
        if right.start[1] < right.end[1]:
            right.start, right.end = right.end, right.start

        self.points = g.PointArray([left.start, left.end, right.end, right.start])

    def selectLines(self):

        goodLines = text.LineCollection()
        for line in self.candidateLines:
            if self.contains(line.box.center.center):
                goodLines.append(line)

        return goodLines

    def contains(self, pointToTest):

        retval = cv2.pointPolygonTest(self.points.numpyArray(), pointToTest, False)

        if retval > 0:
            return True
        else:
            return False

class Margin:

    def __init__(self, lineCollection=None):

        self.left = None        # Each of these are a Line instance
        self.right = None
        self.top = None
        self.bottom = None

        self.height = None
        self.width = None

        self.angle = None

        if lineCollection:
            self.fit(lineCollection)

    def fit(self, lines):

        # Collate all the contours from all the 'border' words (those on the first and last lines, and
        # the first and last words from all other lines).
        borderPoints = g.PointArray()
        for word in self.selectBorderWords(lines):
            for wrappedPoint in word.contour:
                borderPoints.append(wrappedPoint[0])

        self.angle = lines.avgAngle

        # interestingly, it's faster to sort the whole list, which is theoretical O(n log n), than it is
        # to do a min operation followed by a max operation, both of which are O(n).
        sortedHorizontally = sorted(borderPoints, key=lambda point: point.rotate(self.angle).x)
        leftPoint = sortedHorizontally[0]
        rightPoint = sortedHorizontally[-1]

        sortedVertically = sorted(borderPoints, key=lambda point: point.rotate(self.angle).y)
        topPoint = sortedVertically[0]
        bottomPoint = sortedVertically[-1]

        self.left = g.Line([leftPoint], self.angle)
        self.right = g.Line([rightPoint], self.angle)
        self.top = g.Line([topPoint], self.angle+90)
        self.bottom = g.Line([bottomPoint], self.angle+90)

        self.height = abs( topPoint.rotate(self.angle).y - bottomPoint.rotate(self.angle).y )
        self.width  = abs( leftPoint.rotate(self.angle).x - rightPoint.rotate(self.angle).x )

    def selectBorderWords(self, lines):

        borderWords = []
        firstLine = 0
        lastLine = len(lines)-1
        for lineNum, line in enumerate(lines):

            # For the first and last lines, we want to take all the words on the line.
            if lineNum in [firstLine, lastLine]:
                for word in line.words:
                    borderWords.append(word)

            # For all other lines, we can just take the first and last words. This makes it faster.
            else:
                for word in [line.words[0], line.words[-1]]:
                    borderWords.append(word)

        return borderWords

    def paint(self, image, color=colors.BLUE):

        for line in [self.left, self.right, self.top, self.bottom]:
            image = line.paint(image, color)

        #image = self.box.paint(image, color)

        #cv2.fillPoly(image, [self.corners.numpyArray()], colors.BLUE)

        return image
