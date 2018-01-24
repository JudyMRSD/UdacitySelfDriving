# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

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
![alt text][gray_scale]

2.Canny edge detection, output img_edges
![alt text][canny_edge]

3.Use cv2.fillPoly to ignore everything outside region of interst, input: img_edges, output: img_edges_masked
![alt text][region_mask]
4. Hough transform to detect lines in an image, input : img_edges_masked, output: img_lines
![alt text][region_mask]
5. Extrapolate line segments, superimpose on the original image, output as final result
![alt text][hough_edge]



---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.



### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
