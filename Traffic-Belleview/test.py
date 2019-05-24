import cv2
import matplotlib.pyplot as plt

fn = 'input.avi'

cap = cv2.VideoCapture(fn)
print(cap.isOpened())

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	print(ret)
	print(frame.shape)

	plt.imshow(frame)
	plt.pause(0.001)
