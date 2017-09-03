import numpy as np
import cv2

class Filter():
    # Directional Gradient
    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        # Return the result
        return binary_output

    # Gradient magnitude
    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    # Gradient direction
    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output =  np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    '''
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    '''
    def region_of_interest(self, img, vertices):
        #defining a blank mask to start with
        mask = np.zeros_like(img)
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by \"vertices\with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image


    def process(self, image,plot = False):
        # Blur image
        image = cv2.GaussianBlur(image, (5, 5), 0)
        # Convert to HSL colorspace
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # Grayscale image
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Threshold saturation channel
        s = hls[:,:,2]
        s_thresh = np.zeros_like(gray)
        s_thresh[(s > 160) & (s < 255)] = 1
        # Threshold using sobel operation
        x_gradient = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=7, thresh=(10, 255))
        y_gradient = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=7, thresh=(60, 255))
        mag_gradient = self.mag_thresh(gray, sobel_kernel=7, thresh=(40, 255))
        dir_gradient = self.dir_threshold(gray, sobel_kernel=7, thresh=(.65, 1.05))
        # Merge all channels together into one binary image
        merged = np.zeros_like(gray)
        merged[((x_gradient == 1) & (y_gradient == 1)) | ((mag_gradient == 1) & (dir_gradient == 1)) | (s_thresh == 1)] = 1
        # ROI polygon
        left_bottom = (100, 720)
        right_bottom = (1260, 720)
        left_top = (700,480)
        right_top = (650,480)
        inner_left_top = (610, 410)
        inner_right_top = (680, 410)
        inner_left_bottom = (310, 720)
        inner_right_bottom = (1150, 720)
        vertices = np.array([[left_bottom, inner_left_top, inner_right_top, \
                              right_bottom, inner_right_bottom, \
                              left_top, right_top, inner_left_bottom]], dtype=np.int32)

        # Masked area
        masked = self.region_of_interest(merged, vertices)

        if plot:
            import matplotlib.pyplot as plt
            images = [gray,s_thresh,x_gradient,y_gradient,\
                    mag_gradient,dir_gradient,merged,masked]
            titles = ['Original Image', 'Saturation Threshold', 'Gradient X','Gradient Y','Gradient Magnitude',\
                    'Gradient Direction','Merged Result','Masked Result']
            f = plt.figure(figsize=(30, 30))
            for i,(img, title) in enumerate(zip(images,titles)):
                plt.subplot(4,2,i+1)
                plt.imshow(img,cmap="gray")
                plt.title(title,fontsize = 20)
            plt.tight_layout()
            f.savefig("output_images/filter_result.png")
            plt.show()

        return masked


if __name__ == "__main__":
    import os
    # process the image
    img = cv2.imread(os.path.join("test_images",'test6.jpg'))
    fil = Filter()
    fil.process(img,plot=True)
