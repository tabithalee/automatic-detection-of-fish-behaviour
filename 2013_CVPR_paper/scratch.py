import cv2
import numpy as np
import numpy.linalg as LA
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


# TODO - make this applicable to all the video volumes

# find the last usable index
end_x_index = xLen - 2 * block_size
end_y_index = yLen - 2 * block_size
end_z_index = zLen - 2 * block_size

# get gradients for one slice of volume (5x240x320)
x_indices = range(0, end_x_index+1)
y_indices = range(0, end_y_index+1)
z_indices = range(0, end_z_index+1)

Gx = np.zeros(len(z_indices) * len(y_indices) * len(x_indices), dtype=np.float32)
Gy = np.zeros(len(z_indices) * len(y_indices) * len(x_indices), dtype=np.float32)
Gt = np.zeros(len(z_indices) * len(y_indices) * len(x_indices), dtype=np.float32)

# print(x_indices)

#sliceList = []
#histList = []

x_count = 0
y_count = 0
z_count = 0

Phi_bins = 8
Theta_bins = 16

# Bin the gradient vectors into a history of oriented gradients
Phi_range = (-(math.pi / 2), math.pi / 2)
Theta_range = (-math.pi, math.pi)

sliceList = np.zeros((len(y_indices), len(x_indices), Phi_bins+Theta_bins), dtype=np.float32)
# histList = np.zeros((len(z_indices), len(y_indices), len(x_indices), Phi_bins+Theta_bins))

# avoiding dots inside the for loop makes it faster!
norm = LA.norm
myDivide = np.divide
myAtan = np.arctan
myMax = np.max
mySum = np.sum
myEinsum = np.einsum
myHist = np.histogram
myConcat = np.concatenate

# myZ = 0     # one video volume slice for now
for myZ in z_indices:
    for myY in y_indices:
        for myX in x_indices:
            # TODO - may not actually need these for loops but are the histograms for each volume or each slice?
            for z in range(kernelSize):
                for y in range(kernelSize):
                    for x in range(kernelSize):
                        start_z = myZ + z
                        end_z = myZ + z + kernelSize
                        start_y = myY + y
                        end_y = myY + y + kernelSize
                        start_x = myX + x
                        end_x = myX + x + kernelSize
                        my_G_index = myZ * end_x_index + myY * end_y_index + myX
                        myVolume = myArray[start_z: end_z, start_y: end_y, start_x: end_x]

                        Gx[my_G_index] = myEinsum('ijk,ijk->', Cx, myVolume)
                        Gy[my_G_index] = myEinsum('ijk,ijk->', Cy, myVolume)
                        Gt[my_G_index] = myEinsum('ijk,ijk->', Ct, myVolume)
                        # print('myx: ', myX, 'x: ', x, 'size of x_indices: ', len(x_indices), 'last x index: ', x_indices[-1])

            # Convert the vector to polar coordinates according to the paper

            # Get a vector of the euclidean distances
            Gs = norm([Gx, Gy], axis=0)
            # print('Gs shape: ', Gs.shape, 'Gx shape: ', Gx.shape, 'Gy shape: ', Gy.shape, 'Gz shape: ', Gt.shape)

            # do calculation for Gs
            e_max = myMax(Gs) * 0.01

            spatial_sum = mySum(Gs) + e_max

            # Gs /= spatial_sum
            Gs = myDivide(Gs, spatial_sum)

            M = norm([Gs, Gt], axis=0)

            Theta = myAtan(myDivide(Gy, Gx))
            Phi = myAtan(myDivide(Gt, Gs))

            # Find the histogram in the Phi direction (8 bins)
            # unique_list = list(set(Phi))
            # print('unique values in Phi: ', unique_list)

            Phi_hist, _ = np.histogram(Phi, bins=8, range=Phi_range, weights=M, density=True)


            # Find the histogram in the Theta direction (16 bins)
            # unique_list = list(set(Theta))
            # print('unique values in Theta: ', unique_list)

            Theta_hist, _ = myHist(Theta, bins=16, range=Theta_range, weights=M, density=True)

            sliceList[myY, myX] = (myConcat((Phi_hist, Theta_hist)))
            #  print(histList[-1])

            #Gx.clear()
            #Gy.clear()
            #Gt.clear()

            x_count +=1
        y_count += 1
    z_count += 1

    print('x count: ', x_count, 'y count: ', y_count, 'z count: ', z_count)

    #histList[myZ] = sliceList[myY, myX]
    #histList.append(np.reshape(np.array(sliceList), (len(y_indices), len(x_indices), Phi_bins+Theta_bins)))
    # sliceList.clear()


print(sliceList.shape)
# print(histList.shape)

'''
#plt.bar(range(24), [Phi_hist, Theta_hist])
plt.bar(range(24), np.concatenate((Phi_hist, Theta_hist)))
plt.show()
'''