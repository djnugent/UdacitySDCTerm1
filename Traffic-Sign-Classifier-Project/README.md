# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./figures/histogram.png "Histogram"
[orig]: ./figures/orig.png "Original"
[norm0]: ./figures/norm0.png "normalization 0"
[norm1]: ./figures/norm1.png "normalization 1"
[top5]:  ./figures/top5.png
[image1]: ./images/1.jpg "Traffic Sign 1"
[image2]: ./images/2.jpg "Traffic Sign 2"
[image3]: ./images/3.jpg "Traffic Sign 3"
[image4]: ./images/4.jpg "Traffic Sign 4"
[image5]: ./images/5.jpg "Traffic Sign 5"
[image6]: ./images/6.jpg "Traffic Sign 6"
[image7]: ./images/7.jpg "Traffic Sign 7"
[image8]: ./images/8.jpg "Traffic Sign 8"
[image9]: ./images/9.jpg "Traffic Sign 9"
[image10]: ./images/10.jpg "Traffic Sign 10"
[image11]: ./images/11.jpg "Traffic Sign 11"
[image12]: ./images/12.jpg "Traffic Sign 12"
[image13]: ./images/13.jpg "Traffic Sign 13"

## Rubric Points

__Dataset Exploration__

In order to explore the dataset I printed various stats about the dataset(num_images, image dimensions, etc). Additionally I showed various images in order to visualize them and created a histogram of class distribution.

__Design and Test a Model Architecture__

The first thing I did for implementing my solution was to preprocess the images. I tried 3 different types of preprocessing(None, correlated normalization, uncorrelated normalization)

The next step involved creating the model. I modified the LeNet-5 Architecture and made it easy to add and adjust layers for model tweaking.

When it came to training I only looked at the validation accuracy and adjusted my epochs for a good accuracy/training time balance.

__Performance on new images__

For this step I gathered new images from the Google and ran the trained network on them and evaluated the accuracy and top 5 prediction suggestions.


---
## Writeup / README

#### Question 1: Provide a write up and link to code
You're reading it! and here is a link to my [project code](https://github.com/djnugent/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### Question 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is **12630**
* The shape of a traffic sign image is **32x32x3**
* The number of unique classes/labels in the data set is **43**

#### Question 2. Include an exploratory visualization of the dataset.
In order to explore the dataset I printed random images and labels to see what it looks like. I then created a histogram to visualize the class distribution in the training, validation, and test sets(shown below). I was surprises to see they all had similar distributions. The distribution wasn't uniform which isn't idle but got to work with what you have.

![histogram][histogram]

### Design and Test a Model Architecture

#### Question 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

![orig][orig]

From the start I knew I didn't want to do grayscale because that is wasting useful data. So I looked into way to normalize RGB images. Sources online said you should normalize RGB images by dividing all channels in by the sum of all channels on a per pixel basis. So r' = r/(r+b+g); b' = b/(r+b+g); g' = g/(r+b+g); I didn't like this approach because it correlated the data unnecessarily. Despite this I tried it anyway and got poor training results.

![norm0][norm0]

I decided to make my own normalization technique which scaled the image channels based on min and max pixel values over the entire image. This boosted contrast and didn't intentionally correlate the channels.

![norm1][norm1]

A third approach I wanted to try was to normalize in the HSV domain but couldn't get OpenCV to import properly and gave up.

I decided not to augment to the dataset due to time constraints, but if I did I would have rotated, screwed, zoomed, and obstructed images. I believe this would be a good way to help the network generalize better and give me the opportunity to make the dataset more uniform.

#### Question 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		  | 32x32x3 RGB image       |
| Convolution 5x5  	  | 1x1 stride, valid padding, relu activation 	|
| Max pooling	      	| 2x2 stride		          |
| Batch normalization	|         	              |
| Convolution 5x5  	  | 1x1 stride, valid padding, relu activation 	|
| Max pooling	      	| 2x2 stride		          |
| Batch normalization	|         	              |
| Flatten             |                         |
| Fully connected		  | 200 nodes        				|
| Dropout             | 50%  threshold          |
| Fully connected	  	| 100 nodes        				|
| Dropout             | 50%  threshold          |
| Softmax				      | 43 classes     					|



#### Question 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model I used a learning rate of 0.001, 15 epochs, 128 batch size, and Adam optimizer. The only thing I really tuned was the number of epochs and batch size. Everything else seemed good enough for this project

#### Question 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **96.4%**
* validation set accuracy of **96%**
* test set accuracy of **94%**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? **I started with LeNet-5 since it was used in class**
* What were some problems with the initial architecture? **It wasn't large enough and couldn't generalize**
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting. **I started adding layers and changing convolution depth and kernel size. I messed with this until I saw a reasonable performance on the validation set. I ended up adding batch normalization, dropouts to help out with generalizing, increasing convolutional depth since there were more image channels, and adding more perceptrons since there where more classes.**
* Which parameters were tuned? How were they adjusted and why? **No parameters were autotuned, but I ultimately just made a "deeper" LeNet-5 with dropout and batch normalization.**
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model? **The most important thing was not making the network too big because it was having trouble converging quickly. Also the addition of batch normalization and dropouts helped.**



### Test a Model on New Images

#### Question 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I found 13 German traffic signs online:

![alt text][image1] ![alt text][image2] ![alt text][image3]
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13]

These images are interesting because they vary in brightness and in background. Some backgrounds are busy and some are a soild color. Another thing about these images is that their zoom levels and positioning vary. This means the signs are taking up different parts of the image. The images are centered for the most part but some do have a slight offset and zoom. 

#### Question 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield      		| Yield  									|
| 30 Km/h  			| General Caution 				|
| 50 Km/h				| 30 Km/h						      |
| 60 km/h	   	| Bumpy Road							|
| Go Straight or left	| Go Straight or left	|
| 20 Km/h  			| 30 Km/h				         |
| No Entry				| No Entry				      |
| Go Straight or right	| Go Straight or right |
| Keep Right	   	| Keep Right						|
| 100 Km/h  			| 30 Km/h				        |
| General Caution 	| General Caution 				|
| No Passing 			| No Passing 				|
| Priority Road  			| Priority Road 		|

The most common misclassifications occur on the speed signs. Either the speed is wrong or it is completely misclassified.

It was odd, the models predictions changed based on the run even though the images remained the same. I am sure there is a random variable being set each run but that seems like undesirable behavior to me. The test images score was any where from 64% to 84% over multiple runs. This result is significantly lower than the test and validation results. I attribute this to the standards in which the images were cropped/selected. The team that made the original data was most likely following a set of rules when it came to creating the dataset. The images I found online were from multiple sources and were following a different standard. It is likely my network is only used to how the images were cropped/selected in the original dataset. A way to combat this would be to shift, skew, zoom, and rotate the original dataset so the network could generalize better for images external to the original dataset.

#### Question 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

![top5][top5]

The classifier is nearly 100% certain on the items it classifies correctly. When it gets them wrong the certainty drops to 93% or less.

If I had more time I'd love to make a ROC curve for this classifier and mess with cutoff values.
