#!/usr/bin/python

import sys
import os
from page import Page

SHOW_STEPS = True    # change this to false if you just want to see the final output for each page.
SAVE_OUTPUT = False

inputFolder = os.path.join('images')
outputFolder = os.path.join('output')

for filename in os.listdir(inputFolder)[:]:
#for filename in ['page332.jpg', 'page335.jpg']:
    
    inputPath = os.path.join(inputFolder, filename)
    outputPath = os.path.join(outputFolder, filename)

    page = Page(inputPath, SHOW_STEPS)
    
    if SAVE_OUTPUT:
        page.save(outputPath)  # save a copy of what is displayed. Used for getting images for the paper.
    
    page.show((800, 800))

