import cv2
import numpy as np

from hog_functions import get_sobel


def get_mbh_descriptor(prvs, next, num_bins):
    # get optical flow first
    of_frame = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    # get the x and y directions
    gx, gy = get_sobel(of_frame)

    mag, ang = cv2.cartToPolar(gx, gy)

    return mag, ang


def get_hof_descriptor(prvs, next, num_bins):
    # Get Gunnar-Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    # Convert the cartesian flow vectors to polar vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hof_hist = np.histogram(ang, bins=num_bins, density=False)

    return hof_hist
