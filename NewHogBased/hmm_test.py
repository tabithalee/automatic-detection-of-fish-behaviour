import numpy as np
import cv2

from hmm import *


erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

video_string = "myVidString"
num_divisions_w = 3
num_divisions_h = 3
tracked_region_list = [1, 2, 3]
number_of_frames = 3
numBins = 16
my_range = np.arange(0, 2 * np.pi + (2 * np.pi / numBins), 2 * np.pi / numBins)
num_states = 5

trainingSet = []
vid_chi_distance_list = []

'''
for vid in training sequence
'''

hof_vid_hist_list, hog_vid_hist_list, mbh_vid_hist_list = get_hists_from_video_frame(video_string, num_divisions_w,
                                                                                     num_divisions_h,
                                                                                     tracked_region_list,
                                                                                     erosion_kernel, dilation_kernel,
                                                                                     number_of_frames, my_range)

trainingSet.append((hof_vid_hist_list, hog_vid_hist_list, mbh_vid_hist_list))

'''
end for loop
'''

initial_cluster = [trainingSet[i:i+num_states] for i in range(0, len(trainingSet), num_states)]
exemplars = sort_clusters(initial_cluster, num_states)
for item in mbh_vid_hist_list:
    vid_chi_distance_list.append(get_chi_distance_to_every_exemplar(item, exemplars))