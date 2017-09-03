import numpy as np
import cv2
from skimage.feature import hog



def convert_color(image, cspace):
    cvt = image
    if cspace == 'HSV':
        cvt = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
        cvt = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
        cvt = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
        cvt = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCB':
        cvt = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCB)
    return cvt.astype(np.float32)/127 - 1.0

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def imread(image):
    return cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)

def extract_feature(img, orient=9,pix_per_cell=8, cell_per_block=2 , spatial_size=(32, 32), hist_bins=32):

    # Extract HOG for this patch
    hog_feat1 = get_hog_features(img[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog_feat2 = get_hog_features(img[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog_feat3 = get_hog_features(img[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

    # Get color features
    spatial_features = bin_spatial(img, size=spatial_size)
    hist_features = color_hist(img, nbins=hist_bins)

    # Return feature vector
    return np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)[0]
