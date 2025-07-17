import cv2
import numpy as np
image = cv2.imread('image.jpg', 0)
_, binary = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=1)
cv2.imshow('Binary Image', binary)
cv2.imshow('Dilated Image', dilated)
cv2.waitKey(0)
cv2.destroyAllWindows()
