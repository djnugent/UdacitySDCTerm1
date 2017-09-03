
**Finding Lane Lines on the Road**


---

### Reflection

###1. My pipeline has 5 steps:
1. Blur the image
2. Canny edge detection
3. Mask the image where the lanes are likely to be found
4. Hough lines
5. Split the image into left and right halves and apply linear regression to all the points found using hough lines.

The only thing different about my pipeline compared to what we did in class is my pipeline contains a merge_lines() function. This takes in hough line segments and an ROI and merges all the line endpoints into a single line segment using linear regression(numpy)


###2. Identify potential shortcomings with your current pipeline

The biggest shortcoming is robustness. The hyperparamters are extremely sensitive to changes in the environment such as lighting, camera framing, and lane visibility.

Currently my merge_lines() uses a ROI and line angle to filter lines of interest. I wish I could just use line angle because the ROI could break down if the car isn't centered in the lanes.


###3. Suggest possible improvements to your pipeline

I tried to combat the lighting issue(for the challenge) by creating a brightness normalizer(balance() function) in order to boost contrast. This has worked in the past for projects of mine but it didn't help the challenge. I suspect there is another issue that causes my code to crash that I haven't looked into.

I would also like to create a more tuned linear regression algorithm that doesn't require an ROI for merging lines. It would exclude outliners instead. This could help with off center lane placement and require fewer parameters.
