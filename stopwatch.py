import time

class Stopwatch:

    def __init__(self, message=None):

        self.initialised = False

        self.startTime = time.time()
        self.lastLapTime = time.time()

        self.pauseStartTime = None
        self.pauseDuration = 0
        self.totalPauseDurationInRun = 0
        self.runTimes = []

        if message is not None:
            self.lap(message)

    def __getTotalRunTime(self):
        currentTime = time.time()
        return currentTime - self.startTime - self.totalPauseDurationInRun

    def lap(self, message):

        currentTime = time.time()
        lapTime = currentTime - self.lastLapTime - self.pauseDuration

        print "%.2f\t%.2f\t%s" %(self.__getTotalRunTime(), lapTime, message)
        self.lastLapTime = currentTime
        self.pauseStartTime = None
        self.pauseDuration = 0

    def pause(self):

        self.pauseStartTime = time.time()

    def unpause(self):

        if self.pauseStartTime is not None:
            currentTime = time.time()
            self.pauseDuration += currentTime - self.pauseStartTime
            self.totalPauseDurationInRun += currentTime - self.pauseStartTime

        self.pauseStartTime = None

    def endRun(self):

        self.runTimes.append(self.__getTotalRunTime())
        average = sum(self.runTimes) / (len(self.runTimes))
        print "average time: %.2f" %average
        print

        self.pauseStartTime = None
        self.pauseDuration = 0
        self.totalPauseDurationInRun = 0

    def reset(self, message="reset"):

        if not self.initialised:
            self.initialised = True

        self.pauseStartTime = None
        self.pauseDuration = 0
        self.totalPauseDurationInRun = 0

        self.startTime = time.time()
        self.lastLapTime = time.time()
        self.lap(message)
