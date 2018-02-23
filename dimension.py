class Dimension():

    def __init__(self, x, y):
        self.x = abs(x)
        self.y = abs(y)

    def __str__(self):
        return '(x:%i, y:%i)' %(self.x, self.y)

    def scale(self, ratio):
        self.x = int(self.x * ratio)
        self.y = int(self.y * ratio)

    def fitInside(self, boundingDimension):
        if (self.x > boundingDimension.x):
            xratio = float(boundingDimension.x) / float(self.x)
            self.scale(xratio)
        if (self.y > boundingDimension.y):
            yratio = float(boundingDimension.y) / float(self.y)
            self.scale(yratio)

    def __iter__(self):
        class Iterator():
            def __init__(self, source):
                self.source = source
                self.index = 0
            def __iter__(self):
                return self
            def next(self):
                if self.index >= len(self.source):
                    raise StopIteration
                else:
                    self.index += 1
                    return self.source[self.index-1]

        return Iterator((self.x, self.y))
