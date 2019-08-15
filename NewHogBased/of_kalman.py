import numpy as np
import cv2
import math

from processing_methods import *
from hog_functions import get_polar_gradients, get_sobel
from mbh import get_mbh_descriptor


def process_video_frame(video_path, num_divisions_W, num_divisions_H):

    cap = cv2.VideoCapture(video_path)

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    hsv2_list = []
    gx_list = []

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

            # get a list of each window's pixels
            prev_list_roi = get_roi(previous_frame, num_divisions_W, num_divisions_H)
            next_list_roi = get_roi(next_frame, num_divisions_W, num_divisions_H)
            hsv_list_roi = get_roi(hsv[..., 2], num_divisions_W, num_divisions_H)

            # get the OF vectors of each window for processing
            for i in range(len(prev_list_roi)):
                hsv2_list.append(cv2.calcOpticalFlowFarneback(prev_list_roi[i], next_list_roi[i], None,
                                                              0.5, 3, 15, 3, 5, 1.1, 0))

                my_stalling = 1

                # take the average motion of each window by averaging out the gx

            # also get MBH vectors of each window TODO



        else:
            break

    return hsv2_list, len(previous_frame)
