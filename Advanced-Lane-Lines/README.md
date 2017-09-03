# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/distortion_result.png "Undistorted"
[image2]: ./output_images/test_distortion_result.png "Road Transformed"
[image3]: ./output_images/filter_result.png "Binary Example"
[image4]: ./output_images/transform_plot.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/pipeline.png "Output"
[video1]: ./project_video_output.mp4 "Video"

---

### Included files

- `AdvancedLaneFinding.py` The main project file
- `camera.py` Contains class for calibrating camera and perspective transforms
- `filter.py` Contains class for filtering image into a binary image that highlights lane lines
- `lane_detector.py` Contains class for detecting lanes in a binary warped image.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I created a class called `Camera` in `camera.py`. Once a camera object is instantiated you must call `Camera.load_calibration()`. If a calibration file is found it will load the preexisting distortion matrix. If not you must call `Camera.calibrate(board_dimensions,img_directory)`. This will run the calibration process and then save the calibration.

In order to calibrate the camera I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

In order to remove the distortion from an image I use the technique described above. But in code it looks this:
```python
cam = Camera()

# load calibration or calibrate camera
board_dimensions = (9,6)
img_directory = "camera_cal"
if( not cam.load_calibration()):
    print("No calibration found. Calibrating camera...")
    cam.calibrate(board_dimensions, img_directory)
else:
    print("Calibration loaded")

# load image
img = cv2.imread(os.path.join("test_images",'test2.jpg'))
# remove distortion
result = cam.undistort(img)
# show image
cv2.imshow("Undistorted", result)
```
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #76 through #109 in `filter.py`).

I use a total of 5 different thresholding techniques in order to create a binary image that highlights the lanes in the image. But before I do the thresholding I preprocess the images into a more meaningful form. The first thing I do is apply a Gaussian blur to remove noise. Next I make a grayscale version of the image for sobel operations and a HSL for channel thresholding.

The following thresholding methods where used:
1. **Threshold the S channel** in the HSL image. I extracted a saturation value that closely matched the lane's saturation values.
2. **X Gradient** using the sobel operation and a X kernel. This extracted vertical lines.
3. **Y Gradient** using the sobel operation and a Y kernel. This extracted horizontal lines.
4. **Magnitude Gradient** using the sobel operation and a Y and Y kernel. This extracted the magnitude of edges.
5. **Direction Gradient** using the sobel operation and a X and Y kernel. This extracted the edge directions.

The final step was to combine the all binary images into one. This was done using bitwise operations on the images. The bitwise function was as follows:

`merged = (x_gradient & y_gradient) | (mag_gradient & dir_gradient) | s_thresh`

The final step of this process was to remove extra info by extracting a ROI(just like project 1).

The results of this filtering process can be seen below:

![Threshold][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in my `Camera` class in a function called `perspective_transform()`, which appears in lines 63 through 82 in the file `camera.py`.  The `perspective_transform()` function takes in an image (`img`). I chose to hardcode the source and destination points inside the function in the following manner:

```python
img_size = (img.shape[1], img.shape[0])
src = np.float32([[150+430,460],
                  [1150-440,460],
                  [1150,720],
                  [150,720]])

dst = np.float32([[offset, 0],
                  [img_size[0]-offset, 0],
                  [img_size[0]-offset, img_size[1]],
                  [offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 580, 460      | 200, 0        |
| 710, 460      | 1080, 0       |
| 1150, 720     | 1080, 720     |
| 150, 720      | 200, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for my lane fitting can be found in `lane_detector.py` on lines #67 to #154. Here you'll find 2 methods for fitting a lane. Both methods take in a warped binary image the was created using the processes described above. The first method `fit_line()` attempts to fit a lane without knowing any prior info about the lane. It uses 9 vertical slices and a horizontal histogram to identify peaks of intensity which represent the 2 lanes. Once the lanes have been located in these slices a degree 2 polynomial fit is applied to high intensity pixel locations. When the polynomial are plotted over the original image it results in something like this:

![alt text][image5]

The second method that does lane fitting is called `quick_fit()`. It does the same thing as lane fit but uses information from the previous frame to speed up it's search.

I explored the use of convolutions for lane detection, but I achieved better short term result with the histogram approach. I'd be interesting to see how the two compare if I put more time into implementing the convolution approach.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

When it came to calculating curvature of the lane I used a fairly simple method.
1. Calculate 3 points(near,middle,far) on the left and right lane polynomials
2. Average the 2 sets along their X axis. This merged the 6 points into 3 points running down the center of the lane.
3. Convert the points to world space(pixels -> meters)
4. I then did a 3 point circle calculation in order to extract the radius and center of the circle that this polynomial represented, however, I only used the radius.

Originally I calculated the radius for both lanes and then averaged the radiuses together but this yielded poor results if the lane tips accidently curved in or curved out. This was solved by averaging the 2 lanes before calculating curvature. This cancelled out the lane tips curving in or out.

In order to calculate the lane position I calculated the X position where the lane polynomials intersect with the bottom of the image. This gave individual lane positions in pixel relative to the left of the image. I then took the midpoint of the 2 lanes and set them relative to the center of image. This gave me the lane center position in pixels. I then converted this to meters.

The next step was to run sanity checks to filter out any erroneous lane detections. This just checked lane width and verified the lane curvature hadn't changed too much since the last frame.

The final step involved an Exponential moving average in order to smooth out the readings.

Negative radius and position indicates left turn and left lane positioning. Positive radius and position indicates right turn and right lane positioning.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented a `draw()` function in `lane_detector.py`(lines #208 to #236) which renders the results of the pipeline. Here is a result of the pipeline:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

**or**

Youtube

[![Video](https://img.youtube.com/vi/d_sbNx0iSC4/0.jpg)](https://www.youtube.com/watch?v=d_sbNx0iSC4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


The first step was to implement the individual parts from the lectures and understand how they work. This involved turning them into python classes, making them more understandable(to me) and messing around with the parameters. This was by far the longest process. Once all the individual parts were working I put them all together in a complete pipeline.

One of the hardest parts of this project was figuring out the best way to "merge" information. Like merging the 5 thresholded images. Or merging both lanes in a way that provided a clean result. I messed with a few methods of merging and smoothing lane data that were super clever(in my opinion) but failed in strange ways. I finally settle on simpler method which was described in this write up, but I am confident this is not the best method I can come up with.

Areas of improvement include making my pipeline more robust to noise. I'd like to explore more advanced filtering techniques to identify and ignore erroneous lane detections. This could allow my code to generalize to variable road conditions. My code currently struggles once it catches a "bad" lane and then keeps tracking that lane over multiple frames until the entire pipeline falls apart. I'd like to make it so it never catches those bad lanes and if it does, it realizes it and resets tracking.

Overall, my pipeline just needs to be constrained in some parts and loosened in other parts. This involves tuning parameters better and better "defining"  what a good lane is with sanity checks.
