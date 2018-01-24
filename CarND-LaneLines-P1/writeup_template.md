# **Finding Lane Lines on the Road** 

---

### Pipeline

---
[//]: # (Image References)
[gray_scale]: ./result_images/gray_scale.jpg "Grayscale"
[canny_edge]: ./result_images/canny_edge.jpg "Grayscale"
[region_mask]: ./result_images/region_mask.jpg "Grayscale"
[hough_all_edge]: ./result_images/canny_edge.jpg "hough_all_lines"
[hough_edge]: ./result_images/canny_edge.jpg "hough_lines"

**Finding Lane Lines on the Road**
Summary of my implementation:

1.RGB to grayscale, apply Gaussian smoothing to remove noise and easier to find edge: ouput img_blur
![alt text][gray_scale=250x]
2.Canny edge detection, output img_edges
![alt text][canny_edge=250x]
3.Use cv2.fillPoly to ignore everything outside region of interst, input: img_edges, output: img_edges_masked
![alt text][region_mask=250x]
4. Hough transform to detect lines in an image, input : img_edges_masked, output: img_lines
![alt text][region_mask=250x]
5. Extrapolate line segments, superimpose on the original image, output as final result
![alt text][hough_edge=250x]


**Details on draw_two_lines() function**
input: 
all the output lines from hough line detection

parametesr:
set a min and max slope to identify outliers 

Steps:
loop through all the lines 
    Positive slopes belong to left lines, negative slope belong to right lines.
    Find slope (m) and bias (b) for each line
    if the line is not an outlier:
        store the line parameters (m, b) as left line or right line
        
find the mean value for m, b for left and right lines 

Use mean values to find the two end points for each of the left line and right line



### 2. Identify potential shortcomings with your current pipeline

One shortcoming is that it fails for the challenging case, where the lane lines are curved.  
The hough line function only find staight edges and extrapolate it. 

### 3. Suggest possible improvements to your pipeline

There are two many hand tuned parameters, such as the ones for region of interest masking, Canny edge detector, and hough line detection. This make the pipeline not robust. A potential improvement is to use machine learning to learn these parameters.
