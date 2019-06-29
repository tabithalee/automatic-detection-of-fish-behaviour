import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

'''
# can do these later
def divide_frame():

def get_histogram():

def get_hsv_image():
'''

num_divisionsW = 8
num_divisionsH = 6

# returns a list with the regions of  interest
def get_roi(frame, num_divisionsW, num_divisionsH):
    gridDivisionW = np.floor(frame.shape[1] / num_divisionsW).astype(np.int)
    gridDivisionH = np.floor(frame.shape[0] / num_divisionsH).astype(np.int)
    roi_list = [frame[x*gridDivisionW:(x+1)*gridDivisionW, y*gridDivisionH:(y+1)*gridDivisionH] for x in range(gridDivisionW)
                for y in range(gridDivisionH)]
    print(len(roi_list), np.array(roi_list[0]).shape)
    return roi_list


cap = cv2.VideoCapture('/home/tabi/Desktop/automatic-detection-of-fish-behaviour/good_vids/'
                       'BC_POD1_PTILTVIDEO_20110522T114342.000Z_3.ogg')

'''
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
'''

# Create some random colors
color = np.random.randint(0,255,(100,3))

if (cap.isOpened() == False):
    print("Error opening video stream or file")

# fig = plt.figure()
#fig, figarray = plt.subplots(2, 1)

# Take first frame
ret, frame1 = cap.read()
if ret:
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

# add a count flag for the number
frameCount = 0
numberOfFrames = 3
numBins = 16
myRange = np.arange(0, 2 * math.pi + (2*math.pi/numBins), 2*math.pi/numBins)
#print(myRange)
summedHist = np.zeros((numBins,))
savedPlotCount = 0
# Cannot specify a large array as not enough memory
# frameFlowArray = np.zeros(5 * (frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)

#print('flow array shape: ', frameFlowArray.shape)

while(cap.isOpened()):
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # here is where divide the frame into subregions
        print('testing get_roi!')
        get_roi(prvs, num_divisionsW, num_divisionsH)

        #next = cv2.adaptiveThreshold(next, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # adding a little bit of gaussian to try and remove noise
        # next = cv2.GaussianBlur(next, (31, 31), 0)

        # adding otsu's method and morphological operations
        #_, next = cv2.threshold(next, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        #erosionKernel = np.ones((5, 5), np.uint8)
        erosionKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dilationKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        #next = cv2.erode(next, erosionKernel, iterations=3)
        #next = cv2.morphologyEx(next, cv2.MORPH_OPEN, erosionKernel)
        # filling
        # dilation
        #next = cv2.morphologyEx(next, cv2.MORPH_CLOSE, dilationKernel)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        #print('cart to polar max: ', np.max(mag))

        #print(np.max(hsv[...,0]))
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        #frameHist, bins = np.histogram(ang, bins=myRange, weights=mag, density=True)
        #print('normalized max: ', np.max(hsv[..., 2]))

        #bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        hsv[..., 2] = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_OPEN, erosionKernel)
        hsv[..., 2] = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_CLOSE, dilationKernel)
        _, hsv[..., 2] = cv2.threshold(hsv[..., 2], 12, 255, cv2.THRESH_TOZERO)
        #print('processed max: ', np.max(hsv[..., 2]), 'ang range: ', np.min(ang), np.max(ang))
        frameHist, bins = np.histogram(ang, bins=myRange, weights=hsv[..., 2], density=False)

        hsv[..., 0] = ang * 180 / np.pi / 2
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


        # draw a 3x3 grid onto the image
        # get the top points and the left points
        heightDivision, widthDivision = np.floor(frame2.shape[0] / num_divisionsH).astype(np.int), np.floor(
            frame2.shape[1] / num_divisionsW).astype(np.int)
        topPoints = [(i * widthDivision, 0) for i in range(1, num_divisionsW)]
        bottomPoints = [(i * widthDivision, frame2.shape[0]-1) for i in range(1, num_divisionsW)]
        leftPoints = [(0, i * heightDivision) for i in range(1, num_divisionsH)]
        rightPoints = [(frame2.shape[1]-1, i * heightDivision) for i in range(1, num_divisionsH)]
        # draw a line
        for i in range(len(topPoints)):
            #print(topPoints[i])
            cv2.line(bgr, topPoints[i], bottomPoints[i], (0, 255, 0), thickness=3, lineType=8, shift=0)
        for i in range(len(leftPoints)):
            cv2.line(bgr, leftPoints[i], rightPoints[i], (0, 255, 0), thickness=3, lineType=8, shift=0)


        # add the histograms
        if frameCount is numberOfFrames:
            # save histograms to file
            figNameString = '/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/savedHistograms/' \
                            + '{0:08}'.format(savedPlotCount) + '.png'
            plt.subplot(2, 1, 1)
            plt.ylim(0, 700)
            plt.bar(bins[:-1], summedHist, align='edge', width=2*math.pi/numBins)
            #plt.savefig(figNameString)
            plt.clf()

            savedPlotCount += 1
            frameCount = 0
            summedHist = np.zeros((numBins,))
            #print('saved figure', savedPlotCount)

        summedHist += frameHist

        plt.subplot(2, 1, 2)
        plt.imshow(bgr)
        plt.pause(0.001)

        prvs = next

        frameCount += 1

    else:
        break

cap.release()
cv2.destroyAllWindows()
