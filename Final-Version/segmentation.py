import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def run_segmentation(prevFrame, minArea, threshVal, dilationIterations, gaussianSize, bufferLength, lineThick,
                     saveFrames, gaussianFilter, myList, cap, plot_layout,
                     trackedPoints, predictedPoints, kalmanX, kalmanY,
                     repoPath):

    frame_count = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            initFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(initFrame, gaussianFilter, 0)
            frame = cv2.bilateralFilter(frame, 15, 80, 80, 0)

            if prevFrame is False:
                prevFrame = frame

            frameDiff = cv2.absdiff(prevFrame, frame)
            _, threshDiff = cv2.threshold(frameDiff, threshVal, 255, cv2.ADAPTIVE_THRESH_MEAN_C)

            threshDiff = cv2.dilate(threshDiff, None, iterations=dilationIterations)
            contours, _ = cv2.findContours(threshDiff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < minArea:
                    continue

                # Kalman prediction state
                predictionX = kalmanX.predict()
                predictionY = kalmanY.predict()

                predictedPoints.append((predictionX, predictionY))

                # find center of the contour
                myMoment = cv2.moments(c)
                xMom = int(myMoment['m10'] / myMoment['m00'])
                yMom = int(myMoment['m01'] / myMoment['m00'])

                # Check if the centroids are very different from the last one
                trackedPoints.appendleft((xMom, yMom))


                # Kalman update step
                testX = np.array(xMom, dtype=np.float32).reshape((1,))
                testY = np.array(yMom, dtype=np.float32).reshape((1,))
                kalmanX.correct(testX)
                kalmanY.correct(testY)

                print('prediction: ', predictionX, 'measurement: ', testX)

                # Draw bounding box(es)
                (xPos, yPos, width, height) = cv2.boundingRect(c)

                cv2.rectangle(initFrame, (xPos, yPos), (xPos + width, yPos + height), (0, 255, 0), 2)

                # loop through the tracked points
                for i in range(1, len(trackedPoints)):
                    cv2.line(initFrame, trackedPoints[i-1], trackedPoints[i], (0, 0, 255), lineThick)
                    cv2.line(initFrame, predictedPoints[i-1], predictedPoints[i], (0, 255, 0), lineThick)

            plt.figure("main")
            plt.subplot(plot_layout[0, 0])
            plt.title('Segmented Frame')
            plt.imshow(threshDiff)
            plt.subplot(plot_layout[0, 1])
            plt.title('Original Frame')
            plt.imshow(initFrame)
            plt.pause(0.001)

            if saveFrames is True:
                plt.savefig(''.join((repoPath, '/', '%000d' % frame_count, '.png')))

            myList.append(threshDiff)
            time.sleep(0.001)

            frame_count += 1
        # Break the loop
        else:
            break
