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

        if saveToFolder is True:
            if not os.path.exists(''.join((figPath, dirName, '/', videoTitle))):
                os.mkdir(''.join((figPath, dirName, '/', videoTitle)))
            plt.savefig(figNameString)
            plt.clf()

        savedPlotCount += 1

        if saveToFolder is True:
            print('saved figure', savedPlotCount)

        frameCount = 0

    return frameCount, savedPlotCount


def get_first_derivative(array, fps):
    # get symmetric first derivative of array
    step_size = 3 * float(1 / fps)
    derivative = [((array[i+1] - array[i-1]) / (2 * step_size)) for i in range(1, len(array)-1)]
    return derivative