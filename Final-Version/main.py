import numpy as np
import cv2
import math
import os

import matplotlib.pyplot as plt
import argparse as ap

from scipy.stats import skew, kurtosis
from matplotlib.offsetbox import AnchoredText

from processing_methods import *
from stats import find_weighted_skew, find_weighted_kurtosis

from configparser import ConfigParser


def main(videoTitle, startleFrame, extraStartle, tracked_region_list, saveToFolder, dirName,
         num_divisionsW, num_divisionsH, numberOfFrames, numBins, saveHistograms, saveData, repoPath, fps):


    print(tracked_region_list)
    print(videoTitle, 'startleFrame: ', startleFrame, 'extraStartle: ', extraStartle,
          'roi: ', tracked_region_list, 'save?: ', saveToFolder, 'dirName: ', dirName, 'num div X: ', num_divisionsW,
          'num div Y: ', num_divisionsH, 'sample interval: ', numberOfFrames, 'hist bins: ', numBins,
          'saveHistograms?:', saveHistograms, 'saveData?: ', saveData, 'repo Path: ', repoPath, 'fps: ', fps)

    startleFrame = int(startleFrame)

    if extraStartle:
        extraStartle = int(extraStartle)

    # histogram plot layout formatting
    plt.figure("main figure")
    hist_layout = plt.GridSpec(2, 2)

    videoString = ''.join((repoPath, '/good_vids/sablefish/', videoTitle, '.ogg'))

    frameCount = 0
    savedPlotCount = 0
    frameSummedHist = np.zeros((numBins,))

    non_displayed_region = [i for i in range(1, num_divisionsW * num_divisionsH + 1) if (i not in tracked_region_list)]
    myRange = np.arange(0, 2 * math.pi + (2 * math.pi / numBins), 2 * math.pi / numBins)

    # for plotting the y-axis nicely
    ylim_max = 60

    # -----------------------------------START---------------------------------------------------

    # Create some random colors for direction coding
    color = np.random.randint(0,255,(100,3))

    cap = cv2.VideoCapture(videoString)

    # Check if video stream is valid
    if cap.isOpened() is False:
        print("Error opening video stream or file")

    the_max_max = 0
    pixels_width = 0
    pixels_height = 0
    hist_scaling_factor = 0

    skew_list = []
    kurtosis_list = []
    max_list = []

    hist_skew = 0.00
    hist_kurtosis = 0.00
    hist_max = 0.00

    # Take first frame
    ret, frame1 = cap.read()
    if ret:
        pixels_width = np.floor(frame1.shape[1] / num_divisionsW).astype(np.int)
        pixels_height = np.floor(frame1.shape[0] / num_divisionsH).astype(np.int)
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
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            erosionKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dilationKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

            # get optical flow for entire frame
            hsv, myAngles = get_optical_flow(prvs, next, hsv, erosionKernel, dilationKernel)

            # divide optical flow into grid lists
            mag_list = get_roi(hsv[..., 2], num_divisionsW, num_divisionsH)
            ang_list = get_roi(myAngles, num_divisionsW, num_divisionsH)

            # take only the regions of interest
            tracked_mag_list = [mag_list[i-1] for i in tracked_region_list]
            tracked_ang_list = [ang_list[i-1] for i in tracked_region_list]

            # average out the list
            if frameCount is numberOfFrames:
                # calculate and save the histogram metrics
                avg_angle_list = [x / numberOfFrames for x in summed_tracked_ang]
                avg_mag_list = [x / numberOfFrames for x in summed_tracked_mag]

                flat_avg_angle = np.array(avg_angle_list).flatten()
                flat_avg_mag = np.array(avg_mag_list).flatten()
                hist_skew = find_weighted_skew(flat_avg_angle, flat_avg_mag)
                hist_kurtosis = find_weighted_kurtosis(flat_avg_angle, flat_avg_mag)
                hist_max = np.amax(flat_avg_mag) / hist_scaling_factor

                # only take the histogram of averaged vectors
                frameHist = [np.histogram(avg_angle_list[i], bins=myRange,
                                          weights=avg_mag_list[i], density=False)[0]
                             for i in range(len(tracked_region_list))]

                # sum up the histograms per frame
                frameSummedHist = np.sum(frameHist, axis=0)

                # reset the sum back to zero
                summed_tracked_ang = np.zeros((len(tracked_region_list), pixels_height, pixels_width))
                summed_tracked_mag = np.zeros((len(tracked_region_list), pixels_height, pixels_width))

            summed_tracked_mag = [sum(x) for x in zip(summed_tracked_mag, tracked_mag_list)]
            summed_tracked_ang = [sum(x) for x in zip(summed_tracked_ang, tracked_ang_list)]

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # draw grid
            draw_grid(num_divisionsH, num_divisionsW, bgr)

            # not displaying regions that are not of interest
            for i in non_displayed_region:
                non_display_window(bgr, num_divisionsH, num_divisionsW, i)

            plt.figure("main figure")
            plt.subplot(hist_layout[1, 0])

            # match the area of interest
            plt.title('Optical Flow')
            plt.imshow(bgr)

            # display original video
            plt.subplot(hist_layout[1, 1])
            plt.title('Original Video')
            plt.imshow(frame2)
            plt.pause(0.001)

            # plot the histograms
            frameCount, savedPlotCount = plot_histogram(frameCount, savedPlotCount, frameSummedHist, myRange, numBins,
                                                        saveHistograms, hist_layout, dirName, numberOfFrames,
                                                        saveToFolder, videoTitle, ylim_max, hist_scaling_factor,
                                                        repoPath)

            skew_list.append(hist_skew)
            kurtosis_list.append(hist_kurtosis)
            max_list.append(hist_max)

            hist_text = '\n'.join(('skew=%.2f' % hist_skew, 'kurtosis=%.2f' % hist_kurtosis,
                                   'max=%.2f' % hist_max))

            plt.subplot(hist_layout[0, 0:])
            text_box = AnchoredText(hist_text, frameon=True, loc=2, pad=0.15)
            plt.setp(text_box.patch, facecolor='white', alpha=1)
            plt.gca().add_artist(text_box)

            prvs = next
            frameCount += 1

        else:
            break

    # plot the histogram metrics
    plt.figure("data")
    plt.subplot(5, 1, 1, zorder=1)
    # plot startle line
    s1 = plt.axvline(x=startleFrame * numberOfFrames, ymin=-4.4, ymax=1, label='startle', c='red',
                     zorder=20, clip_on=False)
    plt.text(startleFrame * numberOfFrames, max(skew_list) + 0.3, "Startle", color='red')
    if extraStartle:
        s2 = plt.axvline(x=extraStartle * numberOfFrames, ymin=-4.4, ymax=1, label='startle', c='red',
                         zorder=20, clip_on=False)
        plt.text(extraStartle * numberOfFrames, max(skew_list) + 0.3, "Startle", color='red')

    plt.title('Skew')
    p1 = plt.plot(range(len(skew_list)), skew_list, c='blue', zorder=2)

    plt.subplot(5, 1, 4, zorder=-1)
    plt.title('Skew Derivative')
    d1 = get_first_derivative(skew_list, fps)
    plt.plot(range(len(d1)), d1, c='green', zorder=2)

    plt.subplot(5, 1, 2, zorder=-1)
    plt.title('Kurtosis')
    p2 = plt.plot(range(len(kurtosis_list)), kurtosis_list, c='blue', zorder=2)

    plt.subplot(5, 1, 5, zorder=-1)
    plt.title('Kurtosis Derivative')
    d2 = get_first_derivative(kurtosis_list, fps)
    plt.plot(range(len(d2)), d2, c='green', zorder=2)
    plt.xlabel('Saved Plot Frame')

    plt.subplot(5, 1, 3, zorder=-1)
    plt.title('Max')
    p3 = plt.plot(range(len(max_list)), max_list, c='blue', zorder=2)

    plt.subplots_adjust(hspace=1.2)
    plt.tight_layout()

    if saveData is True:
        np.savez(''.join((repoPath, '/savedHistograms/npzfiles/', dirName, '/', videoTitle)), skew_list=skew_list,
                 kurtosis_list=kurtosis_list, d1=d1, d2=d2, max_list=max_list)
        print('saved', dirName, '.npz')

        if saveToFolder is True:
            plt.savefig(''.join((repoPath, '/savedHistograms/',
                        dirName, '/', videoTitle, '.png')))
        else:
            plt.savefig(''.join((repoPath, '/savedHistograms/',
                                 videoTitle, '.png')))

    cap.release()
    cv2.destroyAllWindows()
    plt.close('all')


# only if arguments passed
if __name__ == '__main__':

    parser = ConfigParser()
    parser.read('dev.ini')

    # default settings
    videoTitle = parser.get('single_main_settings', 'videoTitle')
    startleFrame = parser.getint('single_main_settings', 'startleFrame')
    extraStartle = parser.get('single_main_settings', 'extraStartle')
    windows = parser.get('single_main_settings', 'roi')
    saveToFolder = parser.getboolean('single_main_settings', 'saveToFolder')
    dirName = parser.get('single_main_settings', 'dirName')
    num_divisionsW = parser.getint('single_main_settings', 'numDivX')
    num_divisionsH = parser.getint('single_main_settings', 'numDivY')
    numberOfFrames = parser.getint('single_main_settings', 'sampleInterval')
    numBins = parser.getint('single_main_settings', 'numHistBins')
    saveHistograms = parser.getboolean('single_main_settings', 'saveHistograms')
    saveData = parser.getboolean('single_main_settings', 'saveData')
    repoPath = parser.get('user_settings', 'repoPath')
    fps = parser.getint('single_main_settings', 'fps')

    roi = [int(item) for item in windows.split(',')]

    main(videoTitle, startleFrame, extraStartle, roi, saveToFolder, dirName,
         num_divisionsW, num_divisionsH, numberOfFrames, numBins, saveHistograms, saveData, repoPath, fps)
