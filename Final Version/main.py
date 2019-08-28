import numpy as np
import cv2
import math
import os

import matplotlib.pyplot as plt
import argparse as ap

from scipy.stats import skew, kurtosis
from matplotlib.offsetbox import AnchoredText

from utils import segmentation

cap = cv2.VideoCapture(VideoPath)

# TODO - clean up this part
myList = []

# Create a new figure for plt
fig, figarray = plt.subplots(2, 1)
# TODO end


# Check if camera opened successfully
if cap.isOpened() is False:
    print("Error opening video stream or file")

while cap.isOpened():
    if ret:
        run_segmentation()

    else:
        break


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()