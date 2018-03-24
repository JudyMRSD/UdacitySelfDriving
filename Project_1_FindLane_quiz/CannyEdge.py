# Use Canny Edge Detection to find lane lines

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2  #bringing in OpenCV libraries

image = mpimg.imread('exit-ramp.jpg')
plt.imshow(image)

# convert to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
plt.imshow(gray, cmap='gray')

# Define a kernel size for Gaussian smoothing / blurring
# Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size),0)
#  Canny edge detector on this image.
# edges = cv2.Canny(gray, low_threshold, high_threshold)
# applying Canny to the image gray
# and your output will be another image called edges.
# low_threshold and high_threshold are your thresholds for edge detection.
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

# Display
plt.imshow(edges, cmap='Greys_r')

plt.show()