import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('input.jpg', 0)
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title('Original Binary Image')
plt.imshow(binary, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Morphological Gradient')
plt.imshow(gradient, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
