# Dance Dance Revolution (DDR) A3 & Grand Prix Optical Character Recognition (OCR) Tool

This is a mini-project that I made for a friend so that they are able to track EX Score and other DDR A3 Gold/White Cab or Grand Prix data.

The code uses OpenCV and PyTesseract (a python wrapper for tesseract) to scan ROI (regions of interest) and interprets the image text as strings.

These strings are then stored within a csv using Pandas for later viewing.

## Some Caveats

1. The input has to be 1920x1080, the script will automatically upscale/downscale the video/webcam input(if it is 1920x1080) but the OCR accuracy will vary.

2. At this time only the player whose name was entered will be tracked at a time per script. If you have a powerful enough PC you probably can have these two scripts running at once, just make sure that each instance is ran from two different directories.

3. Biggest one of them all, this script will only work for the **A3 version of DDR cabinets**.

4. There are still some bugs. Sue me, hahaha. But don't really, I'll address them asap.

5. Grand Prix for now has to be set to Player 2 for scanning.  
   _Will look into this at a later time_

## Required Installs:

1. [Google's Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)

2. Python 3.X

3. Python Packages needed are listed under requirements.txt  
   _The main script will install these packages automatically if they aren't already installed._

4. Open Broadcaster Software(OBS)

## How to use Script with OBS Virtual Camera:

1. Install OBS, Python, and Tesseract  
   _Make sure to use default settings when install Tesseract_

2. Open OBS, and set the resolution to 1920x1080 and FPS to 30 or 60.

3. Within OBS Virtual Camera settings, select the DDR gameplay source capture and click _Start Virtual Camera_.

4. Open terminal and run _DDR_OCR.py_ python script  
   _The python packages will start installing, if are needed._

5. Select OBS from option in terminal.

6. Select whether you will be scanning A3 or Grand Prix gameplay

7. Enter your player name and enjoy the data capture.

8. Your data is saved in _Scores.csv_.

## How to use Script with Video:

1. Install Python and Tesseract  
   _Make sure to use default settings when install Tesseract_

2. Open terminal and run _DDR_OCR.py_ python script  
   _The python packages will start installing, if are needed._

3. Select Video from option in terminal.

4. Select desired video to scan from within the popup window.

5. Select whether you will be scanning A3 or Grand Prix gameplay

6. Enter your player name and enjoy the data capture.

7. Your data is saved in _Scores.csv_.

## How to script with a Screenshot:

1. Install Python and Tesseract  
   _Make sure to use default settings when install Tesseract_

2. Within OBS Virtual Camera settings, select the DDR gameplay source capture and click _Start Virtual Camera_.

3. Open terminal and run _DDR_OCR_Image.py_ python script  
   _The python packages will start installing, if are needed._

4. Select desired image to scan from within the popup window.

5. Select whether you will be scanning A3 or Grand Prix gameplay

6. Enter your player name and enjoy the data capture.

7. Your data is saved in _Scores.csv_.

## Please use for Recreational Purposes only.

## Help me pay my bills (Donation Page)

[Donation Link](https://www.buymeacoffee.com/axlpena)
