# Dance Dance Revolution (DDR) A3 Optical Character Recognition (OCR) Tool

This is a mini-project that I made for a friend so that they are able to track EX Score and other DDR A3 Gold/White Cab data.  

The code uses OpenCV and PyTesseract (a python wrapper for tesseract) to scan ROI(regions of interest) and interpret the image text as text strings.  

These strings are then stored within a csv using Pandas for later viewing.

##Some of the Caveats

1) The input has to be 1920x1080, one could upscale the video/webcam input but the OCR accuracy will vary.

2) Only one player can be tracked at a time per script.

3) Biggest one of them all, this script will only work for the __A3 version of DDR cabinets__. 

4) There are still some bugs. Sue me. But don't really, I'll address them asap.

## Required Installs:

1) [Google's Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html) 

2) Python 3.X

3) Python Packages needed are listed under requirements.txt  
*The main script will install these packages automatically if they aren't already installed.*

