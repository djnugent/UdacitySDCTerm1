# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).The code file is called P1.ipynb and the writeup template is writeup_template.md 

To meet specifications in the project, take a look at the requirements in the [project rubric](https://review.udacity.com/#!/rubrics/322/view)


---

### Reflection

### 1. My pipeline has 5 steps:
1. Blur the image
2. Canny edge detection
3. Mask the image where the lanes are likely to be found
4. Hough lines
5. Split the image into left and right halves and apply linear regression to all the points found using hough lines.

The only thing different about my pipeline compared to what we did in class is my pipeline contains a merge_lines() function. This takes in hough line segments and an ROI and merges all the line endpoints into a single line segment using linear regression(numpy)


### 2. Identify potential shortcomings with your current pipeline

The biggest shortcoming is robustness. The hyperparamters are extremely sensitive to changes in the environment such as lighting, camera framing, and lane visibility.

Currently my merge_lines() uses a ROI and line angle to filter lines of interest. I wish I could just use line angle because the ROI could break down if the car isn't centered in the lanes.


### 3. Suggest possible improvements to your pipeline

I tried to combat the lighting issue(for the challenge) by creating a brightness normalizer(balance() function) in order to boost contrast. This has worked in the past for projects of mine but it didn't help the challenge. I suspect there is another issue that causes my code to crash that I haven't looked into.

I would also like to create a more tuned linear regression algorithm that doesn't require an ROI for merging lines. It would exclude outliners instead. This could help with off center lane placement and require fewer parameters.
