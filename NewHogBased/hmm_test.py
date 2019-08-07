import numpy as np
import cv2

from hmm import get_hists_from_video_frame


erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

video_string = "myVidString"
num_divisions_w = 3
num_divisions_h = 3
tracked_region_list = [1, 2, 3]
number_of_frames = 3
numBins = 16
my_range = np.arange(0, 2 * np.pi + (2 * np.pi / numBins), 2 * np.pi / numBins)

get_hists_from_video_frame(video_string, num_divisions_w, num_divisions_h, tracked_region_list, erosion_kernel,
                     dilation_kernel, number_of_frames, my_range)
