# combine all 3
# Canny : find edge
# Region masking: only take the center part assuming camera at fixed position
# hough transform: find lane lines
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2



print(cv2.bitwise_and(np.array([100]), np.array([30])))

# Read in and grayscale the image
image = mpimg.imread('exit-ramp.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

# Define a kernel size and apply Gaussian smoothing
kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

# Define our parameters for Canny and apply
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
# edges only contain values 0 or 255
print ("unique edges", np.unique(edges)) # unique [  0 255]

# Create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)
ignore_mask_color = 255

# Define a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
# cv2.fillPoly(img, pts, color)
# mask only has 0 or 255 as value
cv2.fillPoly(mask, vertices, ignore_mask_color)
print ("unique mask", np.unique(mask)) # unique [  0 255]

print("edges")
plt.imshow(edges)
plt.show()

print("mask")
plt.imshow(mask)
plt.show()
# Calculates the per-element bit-wise conjunction of two arrays or an array and a scalar
# 1&1=1，1&0=0，0&1=0，0&0=0
# since both mask and edges only have value 0 or 255, masked_edges have 255 only on pixels
# where mask and edges both have 255
masked_edges = cv2.bitwise_and(edges, mask)
print("masked_edges")
'''
# 0 & 255 = 0
print(edges[500][400:410])  # [0 0 0 0 0 0 0 0 0 0]
print(mask[500][400:410]) # [255 255 255 255 255 255 255 255 255 255]
print(masked_edges[500][400:410]) # [0 0 0 0 0 0 0 0 0 0]

print(cv2.bitwise_and(np.array([100]), np.array([30]))) # 4

print ("unique", np.unique(masked_edges)) # unique [  0 255]
'''

plt.imshow(masked_edges)
plt.show()

# Define the Hough transform parameters
# Make a blank the same size as our image to draw on
rho = 2 # distance resolution in pixels of the Hough grid
theta = np.pi/180 # one degree , angular resolution in radians of the Hough grid
# at least 15 points in image space need to be associated with each line segment
threshold = 15     # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40 #minimum number of pixels making up a line
max_line_gap = 20    # maximum gap in pixels between connectable line segments
line_image = np.copy(image)*0 # creating a blank to draw lines on


# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)

# Create a "color" binary image to combine with line image
color_edges = np.dstack((edges, edges, edges))

# Draw the lines on the edge image
lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
print("line_edges")
plt.imshow(lines_edges)
plt.show()

