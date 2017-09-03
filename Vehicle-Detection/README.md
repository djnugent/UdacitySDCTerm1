**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/data_preview.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/hist.png
[image4]: ./output_images/initial_search.png
[image5]: ./output_images/initial_result_1.png
[image6]: ./output_images/initial_result_2.png
[image7]: ./output_images/sliding_window.gif
[image8]: ./output_images/new_result1.png
[image9]: ./output_images/heatmap.png
[image10]: ./output_images/final.png


---
My project includes the following files:

**detect_cars.py** contains the main pipeline

**train_svm.py** trains the svm

**lesson_functions.py** contains helper functions from class(feature extractors)

**experiment.py** code experimenting with feature extraction

**svm.pkl & scaler.pkl** trained svm


---

### Dataset exploration

In order to understand the dataset and know what I am dealing with, I explored the dataset. The images are 64x64 RGB images. There are 8792 images of cars and 8968 images of non cars. Here is an example:

![alt text][image1]


## Feature Extraction
Once I had the dataset I decided to mess around with feature extraction. In my initial experiments I used the RGB colorspace but settled on the __HSL__ colorspace because it is more resilient to lighting and color variations. This was a pretty big deal. It brought my false positive rate down a lot.
Here is what a detection looked like while I was still in the RGB colorspace(a lot of false positives):

![alt text][image6]

I messed around with 3 different types of features

#### Histogram of Oriented Gradients (HOG)
`src/lesson_functions.py - Line 21`

![alt text][image2]

The hog feature is the most complex feature extracted from the image. It does a great job of picking up on edges and textures patterns (Gradients) that are unique to cars. I tuned most of parameters through the use of the forums and validating my results using SVM test accuracy. I decided to use all the channels of the image because all channels hold valuable information and no reason to throw it out. Additionally, I settled on `orient = 9`, `cell_per_block = 2`, `pix_per_cell = 8`

#### Color Histogram
`src/lesson_functions.py - Line 46`

![alt text][image3]

The color histogram is used to extract color characteristics of the image. Cars have similar color signals(especially in the HSV color space). This is fairly simple feature but it helps. I decided on `32 bins` per channel.

#### Spatial Feature
`src/lesson_functions.py - Line 40`

This is the most basic feature. It is a down sampled version of the original. The image get resized from 64x64 to `16x16` and flattened. This gives the classifier spatial awareness.

#### Final Feature
The HOG, color, and spatial features were all flatten and concatenated. This resulted in a final feature size of 6156

### SVM
`src/train_svm.py - Line 83`

I experimented with a few SVM designs but most attempts were futile(because of computation limits). I really wanted to optimize the parameters for the SVM: kernel type, bandwidth, gamma. The biggest problem was that with only 2 options for kernels(linear vs RBF), 2 value choices for bandwidth and gamma, and 3 fold cross validation I would have to train like 36+ models. This was an issue because scikit's regular SVM isn't optimized like the LinearSVC, so it takes forever to run, and I have dinky laptop. I settled and ended up just optimizing bandwidth for the LinearSVC using GridSearch. But running the full parameter optimization will have to be an experiment for another day when I have more compute and time.

In the end I was able to train an SVM with an accuracy of __99.1%__

### Sliding Window Search
`src/detect_cars.py - Line 40`

The sliding window technique described in class was pretty naive. It was scale invariant and didn't perform well. Here is what it looked like:

![alt text][image4]

It worked but didn't trigger a lot of detections per vehicles:

![alt text][image5]

I decided to implement a sliding window that varied in scale:

![alt text][image7]

It triggers much more:

![alt text][image8]

I had 5 different level scales: `70px`, `120px`, `160px`, `180px`, `240px`. The window count totaled to __224__. All levels had an overlap of 80%. Anything less than 70% overlap and there weren't enough detections. I experimented with smaller and larger levels but they just increased the false positives. The small levels picked up too much "image noise" and the large levels didn't provide a tight bounding box.

I spent a lot of time refining these levels and overlaps. I initially had too few and was trying to reduce false positive rate by restricting the search area and increasing number of levels but reducing window count. This was the opposite of what I ended up doing in the end. I now try to detect as much as possible and filter out the false positives later.

### False Positives
`src/detect_cars.py - Line 112`

In order to filter out false positives I used a heat map filter. Each time a car was detected it would increase the heat of it's detection region by a value of one. As you can see below the hottest parts of the image are cars. False positives exist but they have low heat and can be thresholded out.

![alt text][image9]

 Except I don't threshold individual frames. I run a 80% exponential moving average(EMA) between heatmap frames and then threshold that. This helps because cars will be tracked between frames but false positives usually occur randomly on a frame by frame basis. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

 Here is the result of EMA heatmap thresholding:

 ![alt text][image10]


---

### Video (click image)

[![Video](https://img.youtube.com/vi/sqPkG9bfWRc/0.jpg)](https://youtu.be/sqPkG9bfWRc)

---

### Discussion

#### Problems
I had a lot of problems with false positives. This was resolved by using the HSL colorspace and adding more windows to my search.

#### Improvements
This pipeline isn't very optimized. I do too many colorspace mappings just for the sake of ease of rendering. I also run a hog feature extractor on every sliding window instead of just once per frame.

I experimented with the tensorflow object recognition API and it worked really well(just slow) but is a bit overkill since it has 90 class labels. I would like to try a YOLO or SSD and see how that compares.
