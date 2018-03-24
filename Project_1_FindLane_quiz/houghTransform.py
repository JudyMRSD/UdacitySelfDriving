# Hough Transform to find lane lines
# Do relevant imports
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)

# Define parameters for Canny edge detection and apply
low_threshold = 50
high_threshold = 150
masked_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


# Define the Hough transform parameters
# Make a blank the same size as out image to draw on
rho = 1
theta = np.pi/180
threshold = 1
min_line_length = 10
max_line_gap = 1
line_image = np.copy(image)*0 # create a blank to draw line on

# run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on the blank
# line :  (x1, y1, x2, y2) end points
for line in lines:
    for x1, y1, x2, y2 in line:
        # line(img, firstEndPoint, secondEndPoint, color, thickness)
        cv2.line(line_image, (x1, y1), (x2,y2), (255,0,0), 10)
        #plt.imshow(line_image)
        #plt.show()
# Create a "color" binary image to combine with line_img
# stack masked_edges (width*height) to make a color_edges (width*height*3)
# to match the dimension of line_img (width*height*3)

# dstack: Stack arrays in sequence depth wise (along third axis).
color_edges = np.dstack((masked_edges, masked_edges, masked_edges))
plt.imshow(masked_edges)
plt.show()

# draw the lines on the edge image
# addWeighted:  dst = src1*alpha + src2*beta + gamma
# blend two images, weighted sum of two images
# combo = color_edges * 0.8 + line_image*1 + 0
combo = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
plt.imshow(combo)












