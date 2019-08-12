import numpy as np
import cv2
import math

from processing_methods import *
from hog_functions import get_polar_gradients
from mbh import get_mbh_descriptor


def process_video_frame(video_path):

    cap = cv2.VideoCapture(video_path)

    # Take first frame
    ret, frame1 = cap.read()
    if ret:
        previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

    while cap.isOpened():
        ret, frame2 = cap.read()
        if ret:
            

            next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        else:
            break

    return