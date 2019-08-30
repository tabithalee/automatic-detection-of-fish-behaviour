import cv2
import os

import matplotlib.pyplot as plt

from matplotlib.offsetbox import AnchoredText
from collections import deque

from segmentation import run_segmentation
from lk_test import App
from configparser import ConfigParser

parser = ConfigParser()
parser.read('dev.ini')

# segmentation settings
desktopPath = parser.get('user_settings', 'desktopPath')
videoDirPath = parser.get('segmentation_settings', 'videoDirPath')
video = parser.get('segmentation_settings', 'video')
prevFrame = parser.getboolean('segmentation_settings', 'prevFrame')
minArea = parser.getint('segmentation_settings', 'minArea')
threshVal = parser.getint('segmentation_settings', 'threshVal')
dilationIterations = parser.getint('segmentation_settings', 'dilationIterations')
gaussianSize = parser.getint('segmentation_settings', 'gaussianSize')
bufferLength = parser.getint('segmentation_settings', 'bufferLength')
lineThick = parser.getint('segmentation_settings', 'lineThick')
saveFrames = parser.getboolean('segmentation_settings', 'saveFrames')

myList = []
gaussianFilter = (gaussianSize, gaussianSize)
videoPath = '/'.join((desktopPath, videoDirPath, video))

trackedPoints = deque(maxlen=bufferLength)
predictedPoints = deque(maxlen=bufferLength)

# Create the Kalman Filter
kalmanX = cv2.KalmanFilter(1, 1, controlParams=0)
kalmanY = cv2.KalmanFilter(1, 1, controlParams=0)

cap = cv2.VideoCapture(videoPath)

plot_layout = plt.GridSpec(1, 2)


# Check if camera opened successfully
if cap.isOpened() is False:
    print("Error opening video stream or file")

'''
run_segmentation(prevFrame, minArea, threshVal, dilationIterations, gaussianSize, bufferLength, lineThick, saveFrames,
                 gaussianFilter, myList, cap, plot_layout,
                 trackedPoints, predictedPoints, kalmanX, kalmanY)
'''

App(cap, saveFrames).run()

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()