class EmptyObject:

    def __init__(self):
        pass

greyscale = EmptyObject()
greyscale.BLACK = 0
greyscale.WHITE = 255
greyscale.MID_GREY = 127


WHITE = (255, 255, 255)
LIGHT_GREY = (191, 191, 191)
MID_GREY = (127, 128, 128)
DARK_GREY = (63, 63, 63)
BLACK = (0, 0, 0)

BLUE = (255, 0, 0)
CYAN = (255, 255, 0)

GREEN = (0, 255, 0)
LIME_GREEN = (0, 255, 102)

YELLOW = (0, 255, 255)
BURNT_YELLOW = (0, 223, 255)
ORANGE = (0, 127, 255)

RED = (0, 0, 255)
MAGENTA = (255,0,255)

PURPLE = (191, 0, 191)

class cycle:

    def __init__(self, *colors):

        self.colors = colors
        self.index = 0        # this will continually loop through self.colors

    def next(self):
        color = self.colors[self.index]
        self.index = (self.index+1) % len(self.colors)
        return color

    def __iter__(self, limit=None):
        class Iterator():
            def __init__(self, colors, limit):
                self.colors = colors
                self.index = 0        # this will continually loop through self.colors
                self.limit = limit
                self.iterCounter = 0  # this will continually count upwards (it won't loop around)
            def __iter__(self):
                return self
            def next(self):
                if self.limit != None and self.limit > self.iterCounter:
                    raise StopIteration
                else:
                    color = self.colors[self.index]
                    self.index = (self.index+1) % len(self.colors)
                    self.iterCounter += 1
                    return color

        return Iterator(self.colors, limit)
