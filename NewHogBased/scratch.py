import numpy as np
import cv2
import math
import os

import matplotlib.pyplot as plt
import argparse as ap

from scipy.stats import skew
from scipy.stats import kurtosis
from matplotlib.offsetbox import AnchoredText
from sys import argv

# -----------------------------------METHODS-------------------------------------------------


def find_index_location(bgr, num_divisionsH, num_divisionsW, heightDivision, widthDivision, index):
    # calculate left corner point (x,y) of window
    startX = ((index - 1) % num_divisionsW) * widthDivision
    startY = np.floor((index - 1) / num_divisionsH).astype(np.int) * heightDivision

    return startX, startY


def non_display_window(bgr, num_divisionsH, num_divisionsW, non_displayed_region):
    # calculate height and width of window to not display
    heightDivision, widthDivision = np.floor(bgr.shape[0] / num_divisionsH).astype(np.int), np.floor(
        bgr.shape[1] / num_divisionsW).astype(np.int)

    startX, startY = find_index_location(bgr, num_divisionsH, num_divisionsW, heightDivision, widthDivision,
                                         non_displayed_region)

    # assign window to white in HSV
    bgr[startY:startY + heightDivision, startX:startX + widthDivision] = (255, 255, 255)


def get_prvs_windows(index, roi_list, prvs_window_list):
    prvs_window_list.append(roi_list[index - 1])
    return prvs_window_list


def get_next_window(index, roi_list, next_window_list):
    return next_window_list


# returns a list with the pixels of the regions of  interest
def get_roi(frame, num_divisionsW, num_divisionsH):
    gridDivisionW = np.floor(frame.shape[1] / num_divisionsW).astype(np.int)
    gridDivisionH = np.floor(frame.shape[0] / num_divisionsH).astype(np.int)
    roi_list = [(frame[y*gridDivisionH:(y+1)*gridDivisionH, x*gridDivisionW:(x+1)*gridDivisionW])
                for x in range(num_divisionsW) for y in range(num_divisionsH)]
    return roi_list


# returns a histogram and hsv values for displaying
def get_optical_flow(prvs, next, hsv, erosionKernel, dilationKernel, myRange, saliencyMethod):
    # Get Gunnar-Farneback optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)

    # Convert the cartesian flow vectors to polar vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Convert angle to degrees for plotting
    hsv[..., 0] = ang * 180 / np.pi

    # Normalize the magnitude (intensity) to be within range 0-255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Perform morphological operations
    hsv[..., 2] = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_OPEN, erosionKernel)
    hsv[..., 2] = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_CLOSE, dilationKernel)

    # Threshold image
    _, hsv[..., 2] = cv2.threshold(hsv[..., 2], 12, 255, cv2.THRESH_TOZERO)

    # Get the histogram of the area
    # myHist, _ = np.histogram(ang, bins=myRange, weights=hsv[..., 2], density=False)

    return hsv, ang


# Draw a grid nxm grid onto the image
def draw_grid(num_divisionsH, num_divisionsW, bgr):
    # Get the top points and the left points
    heightDivision, widthDivision = np.floor(bgr.shape[0] / num_divisionsH).astype(np.int), np.floor(
        bgr.shape[1] / num_divisionsW).astype(np.int)
    topPoints = [(i * widthDivision, 0) for i in range(1, num_divisionsW)]
    bottomPoints = [(i * widthDivision, bgr.shape[0] - 1) for i in range(1, num_divisionsW)]
    leftPoints = [(0, i * heightDivision) for i in range(1, num_divisionsH)]
    rightPoints = [(bgr.shape[1] - 1, i * heightDivision) for i in range(1, num_divisionsH)]
    # draw a line
    for i in range(len(topPoints)):
        # print(topPoints[i])
        cv2.line(bgr, topPoints[i], bottomPoints[i], (0, 255, 0), thickness=3, lineType=8, shift=0)
    for i in range(len(leftPoints)):
        cv2.line(bgr, leftPoints[i], rightPoints[i], (0, 255, 0), thickness=3, lineType=8, shift=0)


def plot_histogram(frameCount, savedPlotCount, frameHist, myRange, numBins, save, hist_layout, dirName,
                   numberOfFrames, saveToFolder, videoTitle, ylim_max, hist_scaling_factor):
    if frameCount is numberOfFrames:
        # save histograms to file
        figPath = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/'
        if saveToFolder:
            figNameString = figPath + dirName + '/' + videoTitle + '/{0:08}'.format(savedPlotCount) + '.png'
        else:
            figNameString = figPath + '/{0:08}'.format(savedPlotCount) + '.png'

        plt.figure("main figure")
        plt.subplot(hist_layout[0, 0:])
        plt.ylim(0, ylim_max)
        plt.bar(myRange[:-1], frameHist / hist_scaling_factor, align='edge', width=2 * math.pi / numBins)
        #plt.bar(myRange[:-1], frameHist, align='edge', width=2 * math.pi / numBins)

        # plot visualisation stuff
        plt.title('Histogram of Optical Flow')
        plt.ylabel('Scaled Vector Histogram')
        plt.xlabel('Degrees (rad)')
        plt.grid(True)
        plt.tight_layout()

        if save is True:
            if not os.path.exists(''.join((figPath, dirName, '/', videoTitle))):
                os.mkdir(''.join((figPath, dirName, '/', videoTitle)))
            plt.savefig(figNameString)
            plt.clf()

        savedPlotCount += 1

        if save is True:
            print('saved figure', savedPlotCount)

        frameCount = 0

    return frameCount, savedPlotCount

# -----------------------------------PARAMETERS-----------------------------------------------


def main(videoTitle, startleFrame, extraStartle, tracked_region_list, saveToFolder, dirName):
    print(tracked_region_list)
    print(videoTitle, 'startleFrame: ', startleFrame, 'extraStartle: ', extraStartle,
          'roi: ', tracked_region_list, 'save?: ', saveToFolder, 'dirName: ', dirName)

    num_divisionsW = 2
    num_divisionsH = 2

    # tracked_region_list = range(1, num_divisionsW * num_divisionsH + 1)
    # tracked_region_list = [1, 2, 3, 4]

    ylim_max = 60

    numberOfFrames = 3
    numBins = 16

    # startleFrame = 28
    # extraStartle = False

    # videoTitle = 'BC_POD1_PTILTVIDEO_20110616T171904.000Z_2'

    saveHistograms = True
    # saveToFolder = False
    saveData = True

    # saliency methods according to number
    #   1     FineGrained
    #   2     SpectralResidual
    saliencyMethod = 1
    saliencyOn = False

    # blurring parameters (not used!!)
    gaussianSize = (21, 21)

    # histogram plot layout formatting
    plt.figure("main figure")
    hist_layout = plt.GridSpec(2, 2)



    # -----------------------------------DERIVED VARIABLES---------------------------------------
    videoString = ''.join(('/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/good_vids/sablefish/', videoTitle,
                           '.ogg'))

    frameCount = 0
    savedPlotCount = 0
    frameSummedHist = np.zeros((numBins,))

    non_displayed_region = [i for i in range(1, num_divisionsW * num_divisionsH + 1) if (i not in tracked_region_list)]
    myRange = np.arange(0, 2 * math.pi + (2 * math.pi / numBins), 2 * math.pi / numBins)

    # -----------------------------------START---------------------------------------------------

    # Create some random colors for direction coding
    color = np.random.randint(0,255,(100,3))

    cap = cv2.VideoCapture(videoString)

    # Create an opencv saliency object
    if saliencyMethod is 1:
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    elif saliencyMethod is 2:
        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    else:
        print('Not a valid saliency number')

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

            if saliencyOn is True:
                # Compute saliency
                _, next = saliency.computeSaliency(next)

            #next = cv2.GaussianBlur(next, gaussianSize, 0)

            erosionKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dilationKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

            # here is where divide the frame into subregions
            # roi_list_prvs = get_roi(prvs, num_divisionsW, num_divisionsH)
            # roi_list_next = get_roi(next, num_divisionsW, num_divisionsH)

            # prvs_window_list = [roi_list_prvs[i-1] for i in tracked_region_list]
            # next_window_list = [roi_list_next[i-1] for i in tracked_region_list]

            # get optical flow for entire frame
            hsv, myAngles = get_optical_flow(prvs, next, hsv, erosionKernel, dilationKernel, myRange, saliencyMethod)

            # get individual histograms
            #frameHist = [get_histogram(prvs_window_list[i], next_window_list[i], frame_hsv, erosionKernel, dilationKernel,
                                       #myRange, saliencyMethod)[1]
                         #for i in range(len(tracked_region_list))]

            # divide optical flow into grid lists
            mag_list = get_roi(hsv[..., 2], num_divisionsW, num_divisionsH)
            ang_list = get_roi(myAngles, num_divisionsW, num_divisionsH)

            # take only the regions of interest
            tracked_mag_list = [mag_list[i-1] for i in tracked_region_list]
            tracked_ang_list = [ang_list[i-1] for i in tracked_region_list]

            # average out the list
            if frameCount is numberOfFrames:
                # average the lists
                #summed_tracked_mag = tracked_mag_list * 1.0 / numberOfFrames
                #summed_tracked_ang = tracked_ang_list * 1.0 / numberOfFrames

                # only take the histogram of averaged vectors
                frameHist = [np.histogram(summed_tracked_ang[i] / numberOfFrames, bins=myRange,
                                          weights=summed_tracked_mag[i] / numberOfFrames, density=False)[0]
                             for i in range(len(tracked_region_list))]

                # sum up the histograms per frame
                frameSummedHist = np.sum(frameHist, axis=0)

                # reset the sum back to zero
                summed_tracked_ang = np.zeros((len(tracked_region_list), pixels_height, pixels_width))
                summed_tracked_mag = np.zeros((len(tracked_region_list), pixels_height, pixels_width))

            summed_tracked_mag += np.array(tracked_mag_list)
            summed_tracked_ang += np.array(tracked_ang_list)

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
                                                        saveToFolder, videoTitle, ylim_max, hist_scaling_factor)

            # calculate and save the histogram metrics
            hist_skew = skew(frameSummedHist, bias=True)
            skew_list.append(hist_skew)
            hist_kurtosis = kurtosis(frameSummedHist, bias=True, fisher=False)
            kurtosis_list.append(hist_kurtosis)
            hist_max = np.amax(frameSummedHist)
            max_list.append(hist_max)

            hist_text = '\n'.join(('skew=%.2f' % (hist_skew, ), 'kurtosis=%.2f' % (hist_kurtosis, ),
                                   'max=%.2f' % (hist_max / hist_scaling_factor, )))

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

    # print('savedplotcount: ', savedPlotCount, 'hist_skew count: ', len(skew_list))
    plt.subplot(3, 1, 1, zorder=1)
    s1 = plt.axvline(x=startleFrame, ymin=-3.2, ymax=1, label='startle', c='red', zorder=20, clip_on=False)
    plt.text(startleFrame, max(skew_list) + 0.3, "Startle", color='red')
    if extraStartle:
        s2 = plt.axvline(x=extraStartle, ymin=-3.2, ymax=1, label='startle', c='red', zorder=20, clip_on=False)
        plt.text(extraStartle, max(skew_list) + 0.3, "Startle", color='red')
    plt.title('Skew')
    p1 = plt.plot(range(len(skew_list)), skew_list, c='blue', zorder=2)
    plt.subplot(3, 1, 2, zorder=-1)
    plt.title('Kurtosis')
    p2 = plt.plot(range(len(kurtosis_list)), kurtosis_list, c='blue', zorder=2)
    plt.subplot(3, 1, 3, zorder=-1)
    plt.title('Max')
    p3 = plt.plot(range(len(max_list)), max_list, c='blue', zorder=2)
    #plt.subplot(3, 1, 4, zorder=-1)
    #plt.title('Kurtosis / Skew')
    #test = [float(ai)/float(bi) for ai, bi in zip(kurtosis_list, skew_list)]
    #p3 = plt.plot(range(len(max_list)), test, c='blue', zorder=2)

    plt.xlabel('Saved Plot Frame')
    plt.subplots_adjust(hspace=0.6)

    if saveData is True:
        if saveToFolder is True:
            plt.savefig(''.join(('/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/',
                        dirName, '/', videoTitle, '.png')))
        else:
            plt.savefig(''.join(('/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/',
                                 videoTitle, '.png')))

    cap.release()
    cv2.destroyAllWindows()


# only if arguments passed
if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument("-n", "--name", help='name of video file', type=str)
    parser.add_argument("-s", "--startleFrame", help='frame at which startle occurred', type=int)
    parser.add_argument("-e", "--extraStartle", help='if more than one startle occured', type=int)
    parser.add_argument("-r", "--roi", help='list of numbered frames to track',
                        type=lambda s: [int(item) for item in s.split(',')])
    parser.add_argument("--save2folder", dest='saveToFolder', default=False, action='store_true')
    parser.add_argument("-d", "--dirName", help='main directory name if saving to folder', type=str)
    args = parser.parse_args()

    main(args.name, args.startleFrame, args.extraStartle, args.roi, args.saveToFolder, args.dirName)

'''
    videoTitle = args.name
    startleFrame = args.startleFrame
    extraStartle = args.extraStartle
    tracked_region_list = args.roi
    saveToFolder = args.saveToFolder
    dirName = args.dirName
'''