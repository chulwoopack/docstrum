import cv2
import numpy
import math

import colors

class Angle:

    def __init__(self, guess=None, degrees=None, radians=None, gradient=None):

        self.canonical = None # radians is the 'canonical' representation.

        if guess is not None:
            try:
                #try treating it like an Angle object
                self.radians(guess.radians())
            except:
                # otherwise treat it like a number in degrees
                self.degrees(guess)
        elif radians is not None:
            self.radians(radians)
        elif degrees is not None:
            self.degrees(degrees)
        elif gradient is not None:
            self.gradient(gradient)
        else:
            raise TypeError('Angle() takes at least one argument')

    def radians(self, newVal=None):
        if newVal is not None:
            self.canonical = Angle.sanitize(newVal)
        else:
            return self.canonical

    def degrees(self, newVal=None):
        if newVal is not None:
            rads = math.radians(newVal)
            self.canonical = Angle.sanitize(rads)
        else:
            return math.degrees(self.canonical)

    def gradient(self, newVal=None):
        # gradient = rise / run = tan(radians)
        if newVal is not None:
            rads = math.atan(newVal)
            self.canonical = Angle.sanitize(rads)
        else:
            return math.tan(self.canonical)

    def __add__(self, other):
        other = Angle(other)    # voila, we can now do angle2 = angle1 + 45
        raw = self.radians() + other.radians()
        return Angle(radians=Angle.sanitize(raw))

    def __sub__(self, other):
        other = Angle(other)
        raw = self.radians() - other.radians()
        return Angle(radians=Angle.sanitize(raw))

    @staticmethod
    def sanitize(rads):

        rads = float(rads)

        # put it into the range -pi < x < pi, including accounting for wrap-around
        rads = ((rads + math.pi) % (2*math.pi)) - math.pi

        # Our angles are symmetric. 3*pi/4 is equivalent to -pi/4
        if rads > (math.pi/2):
            rads = rads - math.pi
        elif rads < (-math.pi/2):
            rads = rads + math.pi

        return rads

    @staticmethod
    def average(angles):
        # important: this doesn't do well with angles close to +-90 degrees. Even if they're clustered close
        # to one point, they'll be split into >90 degrees and < 90 degrees sets, and average to zero.
        # This comes from the fact that angles are actually angles (i.e. symmetric), not bearings (directions).

        sumOfRads = 0.0
        for angle in angles:
            sumOfRads += angle.radians()
        rawAverage = sumOfRads / len(angles)
        return Angle(radians=Angle.sanitize(rawAverage))

class PointArray:

    def __init__(self, points=[]):

        self.points = []
        for point in points:
            # make sure that each point is a Point instance. Also allows us to accept a generator.
            self.points.append(Point(point))

    def __str__(self):
        # human-readable output
        strings = [point.__str__() for point in self.points]

        return "[%s]" %(", ".join(strings))

    def __repr__(self):
        # machine-readable output
        return self.__str__()

    def append(self, point):
        self.points.append(Point(point))

    def numpyArray(self):
        return numpy.array([ [list(point.align())] for point in self.points ])

    def __getitem__(self, key):
        return self.points.__getitem__(key)

    def __setitem__(self, key, value):
        self.points.__setitem__(key, value)

    def __delattr__(self, key):
        self.points.__setitem__(key, None)

    def __reversed__(self):
        return self.points.__reversed__()

    def __len__(self):
        return self.points.__len__()

    def __iter__(self):
        return self.points.__iter__()

    def paint(self, image, color):
        for point in  self.points:
            image = point.paint(image, color)
        return image


class Point:

    def __init__(self, foo=None, bar=None):

        try:
            # If foo is an array, use that and ignore bar.
            # Note that this also means that Point(Point(foo, bar)) is harmless
            self.x = foo[0]
            self.y = foo[1]
        except:
            # Otherwise treat foo and bar like two numbers
            self.x = foo
            self.y = bar

        self.isPoint = True     # used to test instance type.

    def align(self):
        # return a new point instance where .x and .y are integers

        return Point(numpy.int0(numpy.around([self.x, self.y])))

    def cv2point(self):

        return tuple(self.align())

    def rotate(self, angle):

        angle = Angle(angle)
        rotatedPoint = Point()
        rotatedPoint.x = self.x*math.cos(-angle.radians()) - self.y*math.sin(-angle.radians())
        rotatedPoint.y = self.x*math.sin(-angle.radians()) + self.y*math.cos(-angle.radians())

        return rotatedPoint

    def __str__(self):
        # human-readable output
        return "(x:%s, y:%s)" %(self.x, self.y)

    def __repr__(self):
        # machine-readable output
        return self.__str__()

    def __getitem__(self, key):
        # this is a hack that allows the object to be treated like a list.
        return [self.x, self.y].__getitem__(key)

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        else:
            raise KeyError('key must be 0 or 1')

    def __delattr__(self, key):
        self.__setitem__(key, None)

    def __reversed__(self):
        return Point(self.y, self.x)

    def __len__(self):
        return 2

    def __iter__(self):

        yield self.x
        yield self.y
        raise StopIteration

    def __add__(self, other):
        result = Point()
        result.x = self.x + other.x
        result.y = self.y + other.y
        return result

    def __sub__(self, other):
        result = Point()
        result.x = self.x - other.x
        result.y = self.y - other.y
        return result

    def paint(self, image, color, diameter=3):
        cv2.circle(image, self.cv2point(), diameter, color, 1, cv2.CV_AA)
        return image

    @staticmethod
    def distance(start, end):

        start = Point(start)
        end = Point(end)

        delta = end - start

        distance = math.sqrt(delta.x**2 + delta.y**2)

        return distance

    @staticmethod
    def midpoint(start, end):

        start = Point(start)
        end = Point(end)

        midpoint = Point()
        midpoint.x = float(start.x + end.y) / 2
        midpoint.y = float(start.x + end.y) / 2

        return midpoint


class Line:

    def __init__(self, points=[], inputAngle=None, frame=None):

        self.frame = frame

        self.start = None
        self.end = None
        self.angle = None
        self.group = None

        if inputAngle != None:
            inputAngle = Angle(inputAngle)
        self.inputAngle = inputAngle

        self.points = PointArray(points)
        self.update()

    def append(self, point):

        self.points.append(point)
        self.update()

    def intersect(self, other):

        if (self.start is None) or (self.end is None):
            raise Exception('The PixelLine is underspecified; it requires at least two points')
        if (other.start is None) or (other.end is None):
            raise Exception('The PixelLine is underspecified; it requires at least two points')

        otherX = float(other.start.x)
        otherY = float(other.start.y)
        otherM = float(other.angle.gradient())

        selfX = float(self.start.x)
        selfY = float(self.start.y)
        selfM = float(self.angle.gradient())

        point = Point()
        point.x = (otherY - selfY + selfM*selfX - otherM*otherX) / (selfM - otherM)
        point.y = selfY + selfM*(point.x - selfX)

        return point

    def update(self):

        if (self.inputAngle is not None) and (len(self.points) >= 1):
            self.lineFromPointAngle()

        elif len(self.points) < 2:
            self.start = None
            self.end = None
            self.angle = None

        elif len(self.points) == 2:
            self.lineFromTwoPoints()

        else:
            self.leastSquaresLine()

        self.clipToFrame()

    def lineFromPointAngle(self):
        # We find the line based on the angle and the first point. Note that in this case, the line
        # is effectively infinite.

        hypotenuse = 4000
        datum = self.points[0]
        angle = self.inputAngle + 90

        offset = Point()
        offset.x = int(hypotenuse * math.cos(angle.radians()))
        offset.y = int(hypotenuse * math.sin(angle.radians()))

        self.start = datum - offset
        self.end = datum + offset
        self.angle = self.inputAngle

    def lineFromTwoPoints(self):
        # This is the only case in which the line has a visible start and end point.

        self.start = self.points[0]
        self.end = self.points[1]
        self.angle = self.calculateAngle(self.start, self.end)

    def leastSquaresLine(self):
        # try to fit a least-squares trend line

        multiplier = 2000
        dx, dy, x0, y0 = cv2.fitLine(self.points.numpyArray(), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)

        self.start = Point(int(x0 - dx*multiplier), int(y0 - dy*multiplier))
        self.end = Point(int(x0 + dx*multiplier), int(y0 + dy*multiplier))
        self.angle = self.calculateAngle(self.start, self.end)

    def calculateAngle(self, start, end):

        rise = float(self.end.y) - float(self.start.y)
        run = float(self.end.x) - float(self.start.x)

        return Angle(radians=math.atan2(rise, run))

    def clipToFrame(self):
        if self.frame is not None:
            rawStart, rawEnd = cv2.clipLine(self.frame, self.start, self.end)
            self.start = Point(rawStart)
            self.end = Point(rawEnd)

    def paint(self, image, color=colors.BLUE):

        if (self.start is None) or (self.end is None):
            raise Exception('The Line is underspecified; it requires at least two points')
        else:
            cv2.line(image, self.start.cv2point(), self.end.cv2point(), color, 1, cv2.CV_AA)

        return image







