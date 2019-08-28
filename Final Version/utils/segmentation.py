


def run_segmentation():

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            initFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.GaussianBlur(initFrame, gaussianSize, 0)
            frame = cv2.bilateralFilter(frame, 15, 80, 80, 0)

            if prevFrame is None:
                prevFrame = frame

            frameDiff = cv2.absdiff(prevFrame, frame)
            # _, frameDiff = saliency.computeSaliency(frameDiff)
            # frameDiff = (frameDiff * 255).astype('uint8')
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

            figarray[0].imshow(threshDiff)
            figarray[1].imshow(initFrame)
            plt.pause(0.001)

            myList.append(threshDiff)
            time.sleep(0.001)

        # Break the loop
        else:
            break