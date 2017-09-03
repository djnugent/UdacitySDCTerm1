import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from lesson_functions import get_hog_features, imread, bin_spatial, color_hist, extract_features
import IPython

# ==============================================================================
# Read dataset
# ==============================================================================
import glob

# Read in car and non-car images
images = glob.iglob('../dataset/**/*.png', recursive=True)
cars = []
notcars = []
for image in images:
    if 'non-vehicles' in image:
        notcars.append(image)
    elif 'vehicles' in image:
        cars.append(image)

print("Found {} car images".format(len(cars)))
print("Found {} not car images".format(len(notcars)))

# Show example data
car = imread(cars[100])
notcar = imread(notcars[100])
fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(car)
plt.title("Car")
plt.subplot(1,2,2)
plt.imshow(notcar)
plt.title("Not Car")
fig.savefig("../output_images/data_preview.png",bbox_inches='tight')
plt.show()


# ==============================================================================
# Run HOG feature extractor
# ==============================================================================

# 0 to 1 scale
img = car.astype(np.float32)/255

# channel extraction
ch1 = img[:,:,0]
ch2 = img[:,:,1]
ch3 = img[:,:,2]


# HOG
orient = 9
pix_per_cell = 8
cell_per_block = 2
# Compute individual channel HOG features for the entire image
fet1,hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False, vis = True)
fet2,hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False, vis = True)
fet3,hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False, vis = True)

print("HOG feature len: {}".format(len(fet1)*3))

# Show result
fig = plt.figure()
plt.subplot(1,4,1)
plt.imshow(car)
plt.title("Original")
plt.subplot(1,4,2)
plt.imshow(hog1)
plt.title("HOG Ch1")
plt.subplot(1,4,3)
plt.imshow(hog1)
plt.title("HOG Ch2")
plt.subplot(1,4,4)
plt.imshow(hog3)
plt.title("HOG Ch3")
fig.savefig("../output_images/hog.png",bbox_inches='tight')
plt.show()


# ========================================================================
# Color hist and spatial binning
# ========================================================================
spatial_size = (32, 32)
hist_bins = 32
# Apply bin_spatial() to get spatial color features
spatial_features = bin_spatial(img, size=spatial_size)
print("Spatial features: {}".format(spatial_features))
print("Spatial feature len: {}".format(len(spatial_features)))

# Apply color_hist() also with a color space option now
hist_features = color_hist(img, nbins=hist_bins)
print("Color feature len: {}".format(len(hist_features)))

# Show result
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(car)
plt.title("Original")
plt.subplot(1,2,2)
plt.bar(np.arange(len(hist_features)), hist_features, align='center', alpha=0.5)
plt.xticks([16,48,80], ["ch1",'ch2','ch3'])
plt.title("Color Histogram")
fig.savefig("../output_images/hist.png",bbox_inches='tight')
plt.show()

# ========================================================================
# Final feature extractor
# ========================================================================
imgs = [cars[100]]
features = extract_features(imgs, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32 ,orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel='ALL')
print("Feature Length: {}".format(len(features[0])))
