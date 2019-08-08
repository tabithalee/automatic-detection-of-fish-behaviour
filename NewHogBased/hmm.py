import numpy as np
import cv2
import math

from processing_methods import *
from hog_functions import get_polar_gradients
from mbh import get_mbh_descriptor


def get_hists_from_video_frame(video_string, num_divisions_w, num_divisions_h, tracked_region_list, erosion_kernel,
                               dilation_kernel, number_of_frames, my_range):

    cap = cv2.VideoCapture(video_string)
    frame_count = 0
    hof_summed_hist_list = []
    hog_summed_hist_list = []
    mbh_summed_hist_list = []

    # Take first frame
    ret, frame1 = cap.read()
    if ret:
        pixels_width = np.floor(frame1.shape[1] / num_divisions_w).astype(np.int)
        pixels_height = np.floor(frame1.shape[0] / num_divisions_h).astype(np.int)
        hist_scaling_factor = len(tracked_region_list) * pixels_height * pixels_width

        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        frame_hsv = np.zeros((pixels_height, pixels_width, 3))
        hsv[..., 1] = 255
        frame_hsv[..., 1] = 255

        summed_tracked_ang = np.zeros((len(tracked_region_list), pixels_height, pixels_width))
        summed_tracked_mag = np.zeros((len(tracked_region_list), pixels_height, pixels_width))

    while cap.isOpened():
        ret, frame2 = cap.read()
        if ret:
            nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # get optical flow for entire frame
            hsv, my_angles = get_optical_flow(prvs, nxt, hsv, erosion_kernel, dilation_kernel)

            # get hog for entire frame
            hog_mags, hog_angles = get_polar_gradients(nxt)

            # get mbh for entire frame
            mbh_mags, mbh_angles = get_mbh_descriptor(prvs, nxt)

            # divide optical flow, hog, mbh into grid lists
            mag_list = get_roi(hsv[..., 2], num_divisions_w, num_divisions_h)
            ang_list = get_roi(my_angles, num_divisions_w, num_divisions_h)

            hog_mag_list = get_roi(hog_mags, num_divisions_w, num_divisions_h)
            hog_ang_list = get_roi(hog_angles, num_divisions_w, num_divisions_h)

            mbh_mag_list = get_roi(mbh_mags, num_divisions_w, num_divisions_h)
            mbh_ang_list = get_roi(mbh_angles, num_divisions_w, num_divisions_h)

            # take only the regions of interest
            tracked_mag_list = [mag_list[i - 1] for i in tracked_region_list]
            tracked_ang_list = [ang_list[i - 1] for i in tracked_region_list]

            hog_tracked_mag_list = [hog_mag_list[i - 1] for i in tracked_region_list]
            hog_tracked_ang_list = [hog_ang_list[i - 1] for i in tracked_region_list]

            mbh_tracked_mag_list = [mbh_mag_list[i - 1] for i in tracked_region_list]
            mbh_tracked_ang_list = [mbh_ang_list[i - 1] for i in tracked_region_list]

            # average out the list
            if frame_count is number_of_frames:
                # calculate and save the histogram metrics
                avg_angle_list = [x / number_of_frames for x in summed_tracked_ang]
                avg_mag_list = [x / number_of_frames for x in summed_tracked_mag]

                flat_avg_angle = np.array(avg_angle_list).flatten()
                flat_avg_mag = np.array(avg_mag_list).flatten()

                hog_avg_angle_list = [x / number_of_frames for x in hog_summed_tracked_ang]
                hog_avg_mag_list = [x / number_of_frames for x in hog_summed_tracked_mag]

                flat_hog_avg_angle = np.array(hog_avg_angle_list).flatten()
                flat_hog_avg_mag = np.array(hog_avg_mag_list).flatten()

                mbh_avg_angle_list = [x / number_of_frames for x in mbh_summed_tracked_ang]
                mbh_avg_mag_list = [x / number_of_frames for x in mbh_summed_tracked_mag]

                flat_mbh_avg_angle = np.array(mbh_avg_angle_list).flatten()
                flat_mbh_avg_mag = np.array(mbh_avg_mag_list).flatten()

                # only take the histogram of averaged vectors
                frame_hist = [np.histogram(flat_avg_angle[i], bins=my_range, weights=flat_avg_mag[i], density=False)[0]
                              for i in range(len(tracked_region_list))]

                hog_frame_hist = [np.histogram(flat_hog_avg_angle[i], bins=my_range, weights=flat_hog_avg_mag[i],
                                               density=False)[0] for i in range(len(tracked_region_list))]

                mbh_frame_hist = [np.histogram(flat_mbh_avg_angle[i], bins=my_range, weights=flat_mbh_avg_mag[i],
                                               density=False)[0] for i in range(len(tracked_region_list))]

                # sum up the histograms per frame
                frame_summed_hist = np.sum(frame_hist, axis=0) / hist_scaling_factor
                hog_frame_summed_hist = np.sum(hog_frame_hist, axis=0) / hist_scaling_factor
                mbh_frame_summed_hist = np.sum(mbh_frame_hist, axis=0) / hist_scaling_factor

                # save to a list
                hof_summed_hist_list.append(frame_summed_hist)
                hog_summed_hist_list.append(hog_frame_summed_hist)
                mbh_summed_hist_list.append(mbh_frame_summed_hist)

                # reset the sum back to zero
                summed_tracked_ang = np.zeros((len(tracked_region_list), pixels_height, pixels_width))
                summed_tracked_mag = np.zeros((len(tracked_region_list), pixels_height, pixels_width))

                hog_summed_tracked_ang = np.zeros((len(tracked_region_list), pixels_height, pixels_width))
                hog_summed_tracked_mag = np.zeros((len(tracked_region_list), pixels_height, pixels_width))

                mbh_summed_tracked_ang = np.zeros((len(tracked_region_list), pixels_height, pixels_width))
                mbh_summed_tracked_mag = np.zeros((len(tracked_region_list), pixels_height, pixels_width))

                frame_count = 0

            summed_tracked_mag = [sum(x) for x in zip(summed_tracked_mag, tracked_mag_list)]
            summed_tracked_ang = [sum(x) for x in zip(summed_tracked_ang, tracked_ang_list)]

            hog_summed_tracked_mag = [sum(x) for x in zip(hog_summed_tracked_mag, hog_tracked_mag_list)]
            hog_summed_tracked_ang = [sum(x) for x in zip(hog_summed_tracked_ang, hog_tracked_ang_list)]

            mbh_summed_tracked_mag = [sum(x) for x in zip(mbh_summed_tracked_mag, mbh_tracked_mag_list)]
            mbh_summed_tracked_ang = [sum(x) for x in zip(mbh_summed_tracked_ang, mbh_tracked_ang_list)]

            prvs = nxt
            frame_count += 1

        else:
            break

    return hof_summed_hist_list, hog_summed_hist_list, mbh_summed_hist_list


def get_averages(initial_cluster, num_states):
    my_averages = [(np.sum(initial_cluster[i], axis=0) / len(initial_cluster[i])) for i in range(num_states)]
    return my_averages


# don't need symmetrical distance - comparing to a base
def get_quadratic_chi_distance_vector(hist1_base, hist2):
    return cv2.compareHist(hist1_base, hist2, cv2.HISTCMP_CHISQR)


def sort_clusters(my_clusters, num_states):
    done_flag = True
    average_cluster = get_averages(my_clusters, num_states)
    for i in range(num_states):
        d1 = get_quadratic_chi_distance_vector(average_cluster[i], my_clusters[i])
        d2 = get_quadratic_chi_distance_vector(average_cluster[(i+1) % num_states], my_clusters[i])

        for x in d1:
            if d1[x] > d2[x]:
                my_clusters[(i+1) % num_states].append(my_clusters[i])
                my_clusters.pop(i)
                done_flag = False

    if done_flag is False:
        sort_clusters(my_clusters, num_states)
    else:
        return average_cluster


def get_chi_distance_to_every_exemplar(frame_hist, average_cluster):
    frame_chi_distance_list = []
    for i in range(len(average_cluster)):
        d = get_quadratic_chi_distance_vector(average_cluster, frame_hist)
        frame_chi_distance_list.append(d)
    return frame_chi_distance_list


# average frame to exemplar distance for each exemplar
def calculate_mu_values(exemplar, vid_chi_distances):
    my_mu_list = []
    for item in exemplar:
        exemplar_distance_sum = 0
        exemplar_norm = np.linalg.norm(item)
        for frame_distance in vid_chi_distances:
            exemplar_distance_sum += frame_distance[exemplar.index(item)]

        my_mu_list.append(exemplar_distance_sum / exemplar_norm)
    return my_mu_list


# probability of an observation
def generate_emission_vector(exemplar, vid_chi_distances):
    emission_vector = []
    mu_list = calculate_mu_values(exemplar, vid_chi_distances)
    for mu in mu_list:
        for frame_distance in vid_chi_distances:
            emission_vector.append((1 / mu) * math.exp(-1 * frame_distance / mu))

    return emission_vector


def estimate_trans_prob_matrix(num_states, exemplars, total_hist_list):
    A = np.zeros((num_states, num_states))
    for i in range(num_states):
        for j in range(num_states):
            sum = 0
            cluster_size = 0
            for vid in total_hist_list:
                


    return


# probability of model moving to another state
def generate_trans_prob_matrix():
    return
