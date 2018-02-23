import numpy
from box import Box
import colors

class BoilerPlate:

    def __init__(self):
        self.pageNum = None
        self.chapterTitle = None    # only one of [.chapterTitle, .bookTitle] will be set -- they're mutually exclusive.
        self.bookTitle = None

    def paint(self, image, color):

        image = self.pageNum.paint(image, color, box=True)
        if self.chapterTitle is not None:
            image = self.chapterTitle.paint(image, color, box=True)
        if self.bookTitle is not None:
            image = self.bookTitle.paint(image, color, box=True)

        return image

class SectionTitle:

    def __init__(self, firstLine=None):

        self.contentType = "SectionTitle"

        if firstLine is None:
            self.lines = []
        else:
            self.lines = [firstLine]

    def append(self, line):

        self.lines.append(line)

    def paint(self, image, color=colors.MAGENTA):

        for line in self.lines:
            image = line.paint(image, color, box=True)
        return image

class Figure:

    def __init__(self):

        self.contentType = "Figure"
        self.image = None
        self.caption = []

    def paint(self, image, color=colors.CYAN):

        image = self.image.paint(image, color, box=True)
        for line in self.caption:
            image = line.paint(image, color, box=True)
        return image

class Paragraph:

    def __init__(self, firstLine=None):

        self.contentType = "Paragraph"

        if firstLine is None:
            self.lines = []
        else:
            self.lines = [firstLine]

    def append(self, line):

        self.lines.append(line)

    def __add__(self, other):
        # designed to be used when adding Content()s together, so that paragraphs which are split over
        # a page can be reconstituted.
        pass

    def __getitem__(self, val):
        return self.lines.__getitem__(val)

    def __len__(self):
        return self.lines.__len__()

    def paint(self, image, color=colors.RED):

        points = []
        for line in self.lines:
            for word in line.words:
                for point in word.contour:
                    points.append(point)
        points = numpy.array(points)    # This needs to have the format [ [[a,b]], [[c,d]] ]
        box = Box(points)
        image = box.paint(image, color)

        for line in self.lines:
            image = line.paint(image, colors.BURNT_YELLOW, centerLine=True)

        return image


class ChapterStart:

    def __init__(self, lines):

        self.contentType = "ChapterStart"
        self.chapterNum = lines.pull()
        self.titleLines = []
        self.quoteLines = []

        while not lines.peekStart().isHorizontalRule:
            self.titleLines.append(lines.pull())
        lines.pull()    # discard the <hr> line.

        while not lines.peekStart().isHorizontalRule:
            self.quoteLines.append(lines.pull())
        lines.pull()    # discard the <hr> line.

    def paint(self, image, color=colors.ORANGE):

        image = self.chapterNum.paint(image, color)
        for line in self.titleLines:
            image = line.paint(image, color, box=True)
        for line in self.quoteLines:
            image = line.paint(image, color, box=True)

        return image

class Content:

    def __init__(self, lines, isChapterStart=False):

        self.lines = lines
        self.content = []

        if isChapterStart:
            chapterStart = ChapterStart(self.lines)
            self.content.append(chapterStart)

        self.stateMachine()

    def stateMachine(self):

        try:
            newLine = self.lines.pull()
        except IndexError:
            return

        if newLine.box.height > 300:
            self.SM_newFigure(newLine)
        elif newLine.isCentered:
            self.SM_sectionTitle(newLine)
        else:
            self.SM_newParagraph(newLine)

    def SM_newFigure(self, line):

        figure = Figure()
        figure.image = line

        try:
            newLine = self.lines.pull()
        except IndexError:
            self.content.append(figure)
            return

        if newLine.isCentered:
            self.SM_addCaptionLine(newLine, figure)
        else:
            self.content.append(figure)
            self.SM_newParagraph(newLine)

    def SM_addCaptionLine(self, line, figure):

        figure.caption.append(line)

        try:
            newLine = self.lines.pull()
        except IndexError:
            self.content.append(figure)
            return

        if newLine.isCentered:
            self.SM_addCaptionLine(newLine, figure)
        else:
            self.content.append(figure)
            self.SM_newParagraph(newLine)

    def SM_newParagraph(self, line):

        paragraph = Paragraph(line)

        try:
            newLine = self.lines.pull()
        except IndexError:
            self.content.append(paragraph)
            return

        if newLine.box.height > 300:
            self.content.append(paragraph)
            self.SM_newFigure(newLine)

        elif newLine.isCentered:
            self.content.append(paragraph)
            self.SM_sectionTitle(newLine)

        elif line.isParagraphEnd or newLine.isParagraphStart:
            self.content.append(paragraph)
            self.SM_newParagraph(newLine)

        elif newLine.isParagraphEnd:
            self.SM_paragraphEnd(newLine, paragraph)

        else:
            self.SM_paragraphBody(newLine, paragraph)

    def SM_paragraphBody(self, line, paragraph):

        paragraph.append(line)

        try:
            newLine = self.lines.pull()
        except IndexError:
            self.content.append(paragraph)
            return

        if newLine.box.height > 300:
            self.content.append(paragraph)
            self.SM_newFigure(newLine)

        elif newLine.isCentered:
            self.content.append(paragraph)
            self.SM_sectionTitle(newLine)

        elif newLine.isParagraphStart:
            self.content.append(paragraph)
            self.SM_newParagraph(newLine)

        elif newLine.isParagraphEnd:
            self.SM_paragraphEnd(newLine, paragraph)

        else:
            self.SM_paragraphBody(newLine, paragraph)

    def SM_paragraphEnd(self, line, paragraph):

        paragraph.append(line)

        try:
            newLine = self.lines.pull()
        except IndexError:
            self.content.append(paragraph)
            return

        if newLine.box.height > 300:
            self.content.append(paragraph)
            self.SM_newFigure(newLine)

        elif newLine.isCentered:
            self.content.append(paragraph)
            self.SM_sectionTitle(newLine)

        else:
            self.content.append(paragraph)
            self.SM_newParagraph(newLine)

    def SM_sectionTitle(self, line):
        sectionTitle = SectionTitle(line)

        try:
            newLine = self.lines.pull()
        except IndexError:
            self.content.append(sectionTitle)
            return

        if newLine.box.height > 300:
            self.content.append(sectionTitle)
            self.SM_newFigure(newLine)
        else:
            self.content.append(sectionTitle)
            self.SM_newParagraph(newLine)

    def paint(self, image):

        for item in self.content:
            image = item.paint(image)

        return image
