# HoG functions
import cv2
import numpy as np


def get_sobel(frame):
    #Gx = np.column_stack((-1 * vector, [0, 0, 0], vector))
    #Gy = np.stack((-1 * vector, [0, 0, 0], vector))

    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=1)

    return gx, gy


def get_polar_gradients(frame):
    gx, gy = get_sobel(frame)
    mag, ang = cv2.cartToPolar(gx, gy)

    return mag, ang


def get_hog_descriptor(frame, num_bins):
    mag, ang = get_polar_gradients(frame)
    hog_hist = np.histogram(ang, bins=num_bins, weights=mag, density=False)

    return hog_hist
