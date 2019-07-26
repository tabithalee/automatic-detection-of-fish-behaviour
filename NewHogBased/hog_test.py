import cv2



# open video and prepare each frame
videoTitle = 'BC_POD_1'
videoString = ''.join(('/home/tabitha/Desktop/automatic-detection-of-fish-behaviour/good_vids/sablefish/', videoTitle,
                           '.ogg'))

cap = cv2.VideoCapture(videoString)

if cap.isOpened() is False:
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # get HoG gradient matrix
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=1)

        # convert to polar coordinates
        mag, ang = cv2.cartToPolar(gx, gy)

        # get take kurtosis and skew

# plot kurtosis and skew at end of video