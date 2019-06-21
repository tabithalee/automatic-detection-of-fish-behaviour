import numpy as np
import cv2
import matplotlib.pyplot as plt


cap = cv2.VideoCapture('/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/good_vids/'
                       'BC_POD1_PTILTVIDEO_20110519T091755.000Z_1.ogg')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

if (cap.isOpened() == False):
    print("Error opening video stream or file")

# fig = plt.figure()

# Take first frame
ret, frame1 = cap.read()
if ret:
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

# add a count flag for the number
frameCount = 0
numberOfFrames = 5
print(frame1.shape)
frameFlowArray = np.zeros(5 * (frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
print('flow array shape: ', frameFlowArray.shape)

while(cap.isOpened()):
    ret, frame2 = cap.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # add the flow arrays to a big array

        # processing step


        plt.imshow(bgr)
        plt.pause(0.001)

        prvs = next

        frameCount += 1

    else:
        break

cap.release()
cv2.destroyAllWindows()