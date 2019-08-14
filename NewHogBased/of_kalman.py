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
                hsv2_list[i] = get_optical_flow(prev_list_roi[i], next_list_roi[i], hsv_list_roi[i],
                                                erosion_kernel, dilation_kernel)

                # get the gx of each OF and MBH window
                gx_list[i] = cv2.threshold(get_sobel(hsv2_list[i])[0], 120, 255, cv2.THRESH_TOZERO)[1]

            # also get MBH vectors of each window TODO



        else:
            break

    return