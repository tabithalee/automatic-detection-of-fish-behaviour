import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# -----------------------------------PARAMETERS-----------------------------------------------
num_divisionsW = 2
num_divisionsH = 2

numberOfFrames = 3
numBins = 16
myRange = np.arange(0, 2 * math.pi + (2*math.pi/numBins), 2*math.pi/numBins)

frameCount = 0
savedPlotCount = 0
summedHist = np.zeros((numBins,))

# -----------------------------------METHODS-------------------------------------------------

# returns a list with the pixels of the regions of  interest
def get_roi(frame, num_divisionsW, num_divisionsH):
    gridDivisionW = np.floor(frame.shape[1] / num_divisionsW).astype(np.int)
    gridDivisionH = np.floor(frame.shape[0] / num_divisionsH).astype(np.int)
    roi_list = [frame[x*gridDivisionW:(x+1)*gridDivisionW, y*gridDivisionH:(y+1)*gridDivisionH] for x in range(num_divisionsW)
                for y in range(num_divisionsH)]
    return roi_list


# returns a list of histograms and hsv values for displaying
def get_histogram(prvs, next, hsv, erosionKernel, dilationKernel, myRange):
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


def plot_histogram(frameCount, numberOfFrames, savedPlotCount, frameHist, summedHist, myRange, numBins):
    if frameCount is numberOfFrames:
        # save histograms to file
        figNameString = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/' \
                        + '{0:08}'.format(savedPlotCount) + '.png'
        plt.subplot(2, 1, 1)
        plt.ylim(0, 700)
        plt.bar(myRange[:-1], summedHist, align='edge', width=2 * math.pi / numBins)
        # plt.savefig(figNameString)
        # plt.clf()

        savedPlotCount += 1
        frameCount = 0
        summedHist = np.zeros((numBins,))
        # print('saved figure', savedPlotCount)

    summedHist += frameHist

    '''
    plt.subplot(2, 1, 2)
    plt.imshow(bgr)
    plt.pause(0.001)
    '''

    return frameCount, savedPlotCount, summedHist


# -----------------------------------START---------------------------------------------------

# Create some random colors for direction coding
color = np.random.randint(0,255,(100,3))

cap = cv2.VideoCapture('/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/good_vids/'
                       'BC_POD1_PTILTVIDEO_20110522T114342.000Z_3.ogg')

# Check if video stream is valid
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Take first frame
ret, frame1 = cap.read()
if ret:
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

# define subregion of interest
subregion_list = [0]

while cap.isOpened():
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        erosionKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilationKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

        # here is where divide the frame into subregions
        roi_list = get_roi(prvs, num_divisionsW, num_divisionsH)

        '''
        # get summed histogram of just the area specified
        for i in subregion_list:
        '''
        # get individual histograms
        hsv, frameHist = get_histogram(prvs, next, hsv, erosionKernel, dilationKernel, myRange)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # draw grid
        draw_grid(num_divisionsH, num_divisionsW, bgr)

        # display the subregions of interest

        # black out a corner region
        heightDivision, widthDivision = np.floor(bgr.shape[0] / num_divisionsH).astype(np.int), np.floor(
            bgr.shape[1] / num_divisionsW).astype(np.int)

        non_displayed_region = 2
        startX = (non_displayed_region % num_divisionsW) * widthDivision
        startY = (non_displayed_region % num_divisionsH) * heightDivision
        bgr[startY:startY + heightDivision,
            startX:startX + widthDivision] = (255, 255, 255)

        plt.subplot(2, 1, 2)

        # match the area of interest
        plt.imshow(bgr)
        plt.pause(0.001)

        # add the histograms
        frameCount, savedPlotCount, summedHist = plot_histogram(frameCount, numberOfFrames, savedPlotCount, frameHist,
                                                                summedHist, myRange, numBins)

        prvs = next
        frameCount += 1

    else:
        break

cap.release()
cv2.destroyAllWindows()
