cosc428-structor...
========

I had an open-ended Computer Vision assignment to complete, and an out-of-copyright book that 
I wanted to turn into an ebook. Conventional OCR engines like Tesseract weren't 
able to accurately recognise the page structure, which led to many transcription errors. If I 
could tell Tesseract to ignore certain regions (like images or repeated headers), then I could
greatly reduce the number of errors in the resulting ebook. Thus: for my assignment, I wrote 
a program that takes an image and uses computer vision magick to determine the page's structure. 
So far, my program can detect and locate:

* lines of text,
* paragraphs,
* section titles,
* images and their associated captions,
* boilerplate like page numbers, and
* chapter titles.

Ain't it grand?

![](https://github.com/chadoliver/structor/raw/gh-pages/analysed.jpg)

Dependencies
============

The project is written in Python 2.7.3 and uses the ```cv2``` library for interacting with openCV. It also uses ```numpy``` for some of the mathematical operations. On windows, the best way to get these dependencies is to install the Python(x,y) suite (https://code.google.com/p/pythonxy/), which combines python with a customisable set of scientific computing libraries.

Program Structure
=================

The program's root is ```main.py```, but this simply iterates through images in a folder and constructs a ```Page``` instance from each image. Thus, the real work happens in ```page.py```.

```page.py``` contains a few utility methods and the ```Page``` class. The constructor calls the appropriate methods in order to determine the logical structure of the page. This structure is stored in three objects: ```self.margin```, ```self.content```, and ```self.boilerplate``` (which contains such non-content text objects as the page number and header).

The ```getBuildingBlocks``` method is responsible for finding words, grouping words into textual lines, discarding marginal noise, and fitting a ```Margin``` instance around the remaining lines. Most of these tasks are preformed by calling other functions.

The ```self.content``` object is found by passing the set of lines to the ```Content()``` constructor. This uses a state machine to group lines into figures, paragraphs, section titles, etc. The ```Content``` class, along with a class for each content type, is found in ```content.py```.

The other files can generally be ignored when trying to understand the program; they are largely just convenience classes which represent page elements (such as points, geometric lines, words, text lines, and boxes), as well as supporting tools such as the ```Stopwatch```.

How to Run the Code
===================

Run ```main.py``` using the python interpreter. This will process each page in ```./images```, and for each page a series of 'snapshot' images will be displayed in order to illustrate the algorithm. To show only the final result for each image, set ```showSteps``` in ```main.py``` to ```False```.


