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
# print(myArray.shape)
zLen = myArray.shape[0]
yLen = myArray.shape[1]
xLen = myArray.shape[2]

num_z_blocks = math.floor(zLen / block_size)
num_y_blocks = math.floor(yLen / block_size)
num_x_blocks = math.floor(xLen / block_size)
num_blocks = num_z_blocks * num_y_blocks * num_x_blocks

# find the last usable index
end_x_index = num_x_blocks * (block_size - 1)
end_y_index = num_y_blocks * (block_size - 1)
end_z_index = num_z_blocks * (block_size - 1)

'''
# deal with all the volumes later - work with only one volume for now

sampled_volume_array = np.empty((num_blocks, 3))

for i in range(num_blocks):
    index = np.unravel_index(i, [num_z_blocks, num_y_blocks, num_x_blocks])
    sampled_volume_array[i] = myArray
    print("index: ", index)
'''

initialIndex = np.unravel_index(0, [num_z_blocks, num_y_blocks, num_x_blocks])
# print(initialIndex)
#print(myArray[initialIndex])
# print(myArray[0][0][0])

kernelSize = 5

''' # print the volume of array from indices
for i in range(5):
    for j in range(5):
        for k in range(5):
            print(myArray[initialIndex[0]+i][initialIndex[1]+j][initialIndex[2]+k], end=" ")
        print('\n')
    print('**********')

print('end of script\n')
'''

# put the volume in a buffer
# print('shape of myArray: ', myArray.shape)
# print(myArray[initialIndex[0]:kernelSize, initialIndex[1]:kernelSize, initialIndex[2]:kernelSize])


# get the gradient of the volume
# get the gradients of the volume in a single frame

A = [1, 4, 6, 4, 1]
B = [2, 1, 0, -1, -2]

Gx = []
Gy = []
Gt = []


kernelSize = len(B)

Cx = np.zeros((kernelSize, kernelSize, kernelSize), dtype=int)
Cy = np.zeros((kernelSize, kernelSize, kernelSize), dtype=int)
Ct = np.zeros((kernelSize, kernelSize, kernelSize), dtype=int)

frontFace = np.einsum('i,j->ij', A, A)

for i in range(kernelSize):
    Cx[:, :, i] = B[i] * frontFace

for i in range(kernelSize):
    Cy[:, i, :] = B[i] * frontFace

for i in range(kernelSize):
    Ct[i, :, :] = B[i] * frontFace

# print(frontFace)



for z in range(kernelSize):
    for y in range(kernelSize):
        for x in range(kernelSize):
            Gx.append(np.einsum('ijk,ijk->', Cx, myArray[x:x+kernelSize, x:x+kernelSize, x:x+kernelSize]))
            Gy.append(np.einsum('ijk,ijk->', Cy, myArray[y:y+kernelSize, y:y+kernelSize, y:y+kernelSize]))
            Gt.append(np.einsum('ijk,ijk->', Ct, myArray[z:z+kernelSize, z:z+kernelSize, z:z+kernelSize]))

# Convert the vector to polar coordinates according to the paper

# Get a vector of the euclidean distances
Gs = np.linalg.norm([Gx, Gy], axis=0)

# do calculation for Gs
e_max = max(Gs) * 0.01
spatial_sum = sum(Gs) + e_max

Gs /= spatial_sum

print('Gx[0]: ', Gx[0], ' Gy[0]: ', Gy[0], 'spatial_sum[0]: ', Gs[0])
print('Gx[1]: ', Gx[1], ' Gy[1]: ', Gy[1], 'spatial_sum[1]: ', Gs[1])
print('Gx[2]: ', Gx[2], ' Gy[2]: ', Gy[2], 'spatial_sum[2]: ', Gs[2])