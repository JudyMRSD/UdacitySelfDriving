# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**
Summary of my implementation:

1.RGB to grayscale, apply Gaussian smoothing to remove noise and easier to find edge: ouput img_blur

[image1]: ./result_images/gray_scale.jpg "Grayscale"
2.Canny edge detection, output img_edges
[image1]: ./result_images/canny_edge.jpg "Grayscale"

3.Use cv2.fillPoly to ignore everything outside region of interst, input: img_edges, output: img_edges_masked
[image1]: ./result_images/region_mask.jpg "Grayscale"
4. Hough transform to detect lines in an image, input : img_edges_masked, output: img_lines
[image1]: ./result_images/canny_edge.jpg "hough_all_lines"

5. Extrapolate line segments, superimpose on the original image, output as final result


[//]: # (Image References)

[image1]: ./result_images/hough_lines.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text][image1]


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
