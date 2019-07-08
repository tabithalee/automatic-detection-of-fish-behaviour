import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


# -----------------------------------PARAMETERS-----------------------------------------------


num_divisionsW = 2
num_divisionsH = 2

tracked_region_list = [3, 4]

ylim_max = 60

numberOfFrames = 3
numBins = 16
myRange = np.arange(0, 2 * math.pi + (2 * math.pi / numBins), 2 * math.pi / numBins)

frameCount = 0
savedPlotCount = 0
summedHist = np.zeros((numBins,))

videoString = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/good_vids/' \
              'BC_POD1_PTILTVIDEO_20110522T114342.000Z_3.ogg'

saveHistograms = False

# saliency methods according to number
#   1     FineGrained
#   2     SpectralResidual
saliencyMethod = 2
saliencyOn = False

# blurring parameters
gaussianSize = (21, 21)

# histogram plot layout
hist_layout = plt.GridSpec(2, 2)

# -----------------------------------DERIVED VARIABLES---------------------------------------


non_displayed_region = [i for i in range(1, num_divisionsW * num_divisionsH + 1) if (i not in tracked_region_list)]


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
    roi_list = [(frame[y*gridDivisionH:(y+1)*gridDivisionH, x*gridDivisionW:(x+1)*gridDivisionW]) for x in range(num_divisionsW)
                for y in range(num_divisionsH)]
    return roi_list


# returns a histogram and hsv values for displaying
def get_histogram(prvs, next, hsv, erosionKernel, dilationKernel, myRange, saliencyMethod):
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
    myHist, _ = np.histogram(ang, bins=myRange, weights=hsv[..., 2], density=False)

    return hsv, myHist


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


def plot_histogram(frameCount, numberOfFrames, savedPlotCount, frameHist, summedHist, myRange, numBins,
                   save, hist_layout):

    if frameCount is numberOfFrames:
        # save histograms to file
        figNameString = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/' \
                        + '{0:08}'.format(savedPlotCount) + '.png'
        plt.subplot(hist_layout[0, 0:])
        plt.ylim(0, ylim_max)
        plt.bar(myRange[:-1], summedHist / hist_scaling_factor, align='edge', width=2 * math.pi / numBins)

        # plot visualisation stuff
        plt.title('Histogram of Optical Flow')
        plt.ylabel('Scaled Vector Histogram')
        plt.xlabel('Degrees (rad)')
        plt.grid(True)
        plt.tight_layout()

        if save is True:
            plt.savefig(figNameString)
            plt.clf()

        savedPlotCount += 1
        frameCount = 0
        summedHist = np.zeros((numBins,))
        if save is True:
            print('saved figure', savedPlotCount)

    summedHist += frameHist

    return frameCount, savedPlotCount, summedHist


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
        roi_list_prvs = get_roi(prvs, num_divisionsW, num_divisionsH)
        roi_list_next = get_roi(next, num_divisionsW, num_divisionsH)

        prvs_window_list = [roi_list_prvs[i-1] for i in tracked_region_list]
        next_window_list = [roi_list_next[i-1] for i in tracked_region_list]

        # get HSV for entire frame
        hsv, _ = get_histogram(prvs, next, hsv, erosionKernel, dilationKernel, myRange, saliencyMethod)

        # get individual histograms
        frameHist = [get_histogram(prvs_window_list[i], next_window_list[i], frame_hsv, erosionKernel, dilationKernel,
                                   myRange, saliencyMethod)[1]
                     for i in range(len(tracked_region_list))]

        # sum up the histograms per frame
        frameSummedHist = np.sum(frameHist, axis=0)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # draw grid
        draw_grid(num_divisionsH, num_divisionsW, bgr)

        # not displaying regions that are not of interest
        for i in non_displayed_region:
            non_display_window(bgr, num_divisionsH, num_divisionsW, i)

        plt.subplot(hist_layout[1, 0])

        # match the area of interest
        plt.title('Optical Flow')
        plt.imshow(bgr)

        # display original video
        plt.subplot(hist_layout[1, 1])
        plt.title('Original Video')
        plt.imshow(frame2)
        plt.pause(0.001)

        # add the histograms
        frameCount, savedPlotCount, summedHist = plot_histogram(frameCount, numberOfFrames, savedPlotCount,
                                                                frameSummedHist, summedHist, myRange, numBins,
                                                                saveHistograms, hist_layout)

        # find the maximum value of the histogram
        if np.amax(summedHist) > the_max_max:
            the_max_max = np.amax(summedHist)
            print('new max: ', np.max(summedHist) / hist_scaling_factor)

        prvs = next
        frameCount += 1

    else:
        break

cap.release()
cv2.destroyAllWindows()
