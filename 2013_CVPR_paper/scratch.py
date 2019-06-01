import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import math

from array import *

cap = cv2.VideoCapture('/home/tabitha/Downloads/Traffic-Belleview/input.avi')
myList = []

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

#Create a new figure for plt
#fig = plt.figure()

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        #print('frameshape: ', frame.shape)
        # Display the resulting frame
        #cv2.imshow('Frame', frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        myList.append(frame)
        #plt.imshow(frame)
        #plt.pause(0.001)
        #plt.show()  #blocking, will wait for you to close it
        time.sleep(0.001)

        # Press Q on keyboard to  exit
        k = cv2.waitKey(25)
        # print(k)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print("you pressed q!")
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# convert list to array
myArray = np.array(myList)

# GET THE BLOCK POSITIONS (assuming no overlaps)
# TODO - should be overlapping STVs

block_size = 5

# find the dimensions of the total volume
print(myArray.shape)
zLen = myArray.shape[0]
yLen = myArray.shape[1]
xLen = myArray.shape[2]

num_z_blocks = math.floor(zLen / block_size)
num_y_blocks = math.floor(yLen / block_size)
num_x_blocks = math.floor(xLen / block_size)
num_blocks = num_z_blocks * num_y_blocks * num_x_blocks

'''
# deal with all the volumes later - work with only one volume for now

sampled_volume_array = np.empty((num_blocks, 3))

for i in range(num_blocks):
    index = np.unravel_index(i, [num_z_blocks, num_y_blocks, num_x_blocks])
    sampled_volume_array[i] = myArray
    print("index: ", index)
'''

initialIndex = np.unravel_index(0, [num_z_blocks, num_y_blocks, num_x_blocks])
print(initialIndex)
print(myArray[initialIndex])
print(myArray[0][0][0])

# print the volume of array from indices
for i in range(5):
    for j in range(5):
        for k in range(5):
            print(myArray[initialIndex[0]+i][initialIndex[1]+j][initialIndex[2]+k], end=" ")
        print('\n')
    print('**********')

print('end of script\n')
