import numpy as np
import cv2
import matplotlib.pyplot as plt


class LaneDetector():
    def __init__(self):
        self.last_lane_info = {"radius":None,"position":None}
        self.last_fit = {"left":None,"right":None}
        self.confidence = 50
        self.histogram = None

    def process(self,img):
        # Fit Left and right side
        current_fit = {"left":None,"right":None}
        for side in ["left","right"]:
            # Try to quick fit that lane based on last frame
            quick_fit = self.fit_lane_quick(img,self.last_fit[side],margin=30)
            # Do a full fit when quick fit fails
            if quick_fit is None:
                current_fit[side] = self.fit_lane(img,side,margin=80,minpix=40)
            else:
                current_fit[side] = quick_fit

        # Attempt to calculate lane curvature and position on new image5
        # If we can't we fall back on the last frame(if we have it)
        radius = None
        position = None
        if current_fit["left"] is not None and current_fit["right"] is not None:
            lane_info = self.calc_curvature_and_position(current_fit,img.shape,self.last_lane_info["radius"])
            if lane_info is not None:
                # Detected lane, boost confidence
                self.confidence = min(self.confidence + 5,100)
                # Exponential moving average
                if self.last_lane_info['radius'] is not None:
                    lane_info["radius"] =( 0.3 * lane_info["radius"]) + (0.7 * self.last_lane_info["radius"])
                    lane_info["position"] = (0.3 * lane_info["position"]) + (0.7 * self.last_lane_info["position"])
                # Update previous state
                self.last_fit = current_fit
                self.last_lane_info = lane_info
                radius = lane_info["radius"]
                position = lane_info["position"]
            else:
                # fall back on last frame
                current_fit = self.last_fit

        #unable to detect a lane in this frame
        if radius is None and position is None:
            self.confidence = max(self.confidence - 5,0)
            radius = self.last_lane_info["radius"]
            position = self.last_lane_info["position"]

        # reset state when we have no idea where the lane is
        if self.confidence == 0:
            self.last_fit = {"left":None,"right":None}

        return {"radius":radius,"position":position, "confidence":self.confidence,\
                        "left_fit":current_fit["left"],"right_fit":current_fit["right"]}


    # Fit lane line to image having no previous knowledge of the lane position
    # img: Binary warped image
    # side: left or right lane
    # nwindows: number of vertical windows
    # margin: width of the windows +/- margin
    # minpix: minimum number of pixels found to recenter window
    def fit_lane(self,img,side,nwindows=9,margin=100,minpix=50):

        #check if we have already created a histogram for this frame
        if self.histogram is None:
        # Take a histogram of the bottom half of the image
            self.histogram = np.sum(img[int(img.shape[0]/2):,:], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(self.histogram.shape[0]/2)
        if side == "left":
            x_base = np.argmax(self.histogram[:midpoint])
        if side == "right":
            x_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(img.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = x_base
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin
            # Draw the windows on the visualization image
            #cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            #cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        #if we didn't detect enough points
        if(len(x) < 3):
            return None

        # Fit a second order polynomial
        fit = np.polyfit(y, x, 2)
        return fit


    # Skip the sliding window if we detected lanes in the last frame
    # img: binary_warped image
    # prev_fit: coefficients of a previous lane fit
    # margin: width of the windows +/- margin
    # return None if no previous fit was given
    # return fit coefficients if a lane was successfully fit
    def fit_lane_quick(self,img,prev_fit, margin = 100):
        #No previous lane provided. Can't do quick fit
        if prev_fit is None:
            return None

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #find lane
        lane_inds = ((nonzerox > (prev_fit[0]*(nonzeroy**2) + prev_fit[1]*nonzeroy + prev_fit[2] - margin)) & (nonzerox < (prev_fit[0]*(nonzeroy**2) + prev_fit[1]*nonzeroy + prev_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        #if we didn't detect enough points
        if(len(x) < 3):
            return None

        # Fit a second order polynomial to each
        fit = np.polyfit(y, x, 2)
        return fit

    # Calculate turing radius and position of the lane_width_thresh
    # fit: polynomial fit on left and right lanes
    # img_size: image shape
    # last_radius: last detected lane radius
    # lane_width_thresh: estimated size of the lane
    # radius_thresh: how much the lane radius can change since last frame
    def calc_curvature_and_position(self, fit,img_size,last_radius, lane_width_thresh=(5,7), radius_thresh=3):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/405 # meters per pixel in y dimension
        xm_per_pix = 3.7/500 # meteres per pixel in x dimension

        # Caculate points along curve
        y = np.array([0,img_size[0]/2,img_size[0]],np.float64)
        x = np.zeros(3,np.float64)
        for i,y_pnt in enumerate(y):
            left_x = fit["left"][0]*y_pnt**2 + fit["left"][1]*y_pnt + fit["left"][2]
            right_x = fit["right"][0]*y_pnt**2 + fit["right"][1]*y_pnt + fit["right"][2]
            x[i] = (left_x + right_x)/2

        # convert to world space
        y *= ym_per_pix
        x *= xm_per_pix

        #Caculate direction of curvature
        dir = np.abs(x[0] - x[2]) / (x[0] - x[2])

        # Calculate radius of curve
        m1 = (y[1]-y[0])/(x[1]-x[0])
        m2 = (y[2]-y[1])/(x[2]-x[1])
        xc = (m1*m2*(y[0]-y[2])+m2*(x[0]+x[1])-m1*(x[1]+x[2]))/(2*(m2-m1))
        yc = -(xc-(x[0]+x[1])/2)/m1+(y[0]+y[1])/2
        radius = np.sqrt((x[1]-xc)*(x[1]-xc)+(y[1]-yc)*(y[1]-yc)) * dir
        # radius sanity check
        if last_radius is not None and np.abs(np.log(last_radius)-np.log(radius)) > radius_thresh:
            return None

        # Calculate lane position
        bottom = img_size[0]
        left_lane_pos = (fit["left"][0]*bottom**2 + fit["left"][1]*bottom + fit["left"][2])
        right_lane_pos = (fit["right"][0]*bottom**2 + fit["right"][1]*bottom + fit["right"][2])

        #lane width sanity check
        lane_width = (right_lane_pos - left_lane_pos) * xm_per_pix
        if not (lane_width_thresh[0] < lane_width < lane_width_thresh[1]):
            return None

        #Calculate lane position
        position = img_size[1]/2 -(right_lane_pos + left_lane_pos)/2
        position *= xm_per_pix
        return {"radius":radius, "position":position}

    # Visualization
    def draw(self,img, lane_info, inv_M):
        #check no for lane detection
        if lane_info["left_fit"] is None or lane_info["right_fit"] is None:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'Searching for lane...' ,(500,360), font, 1,(255,0,0),2)
            return img

        #else draw image
        out_img = np.zeros_like(img).astype(np.uint8)
        # Recast the x and y points into usable format for cv2.fillPoly()
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
        left_fitx = lane_info["left_fit"][0]*ploty**2 + lane_info["left_fit"][1]*ploty + lane_info["left_fit"][2]
        right_fitx = lane_info["right_fit"][0]*ploty**2 + lane_info["right_fit"][1]*ploty + lane_info["right_fit"][2]

        pts_left = np.array(np.transpose(np.vstack([left_fitx, ploty])))
        pts_right = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))
        pts = np.vstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(out_img, np.int_([pts]), [0,255, 0])
        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        unwarped = cv2.warpPerspective(out_img, inv_M, (img.shape[1],img.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(img, 1, unwarped, 0.3, 0)
        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,'Turning Radius: {:2.1f} KM'.format(lane_info["radius"]/1000) ,(10,30), font, 1,(255,255,255),2)
        cv2.putText(result,'Lane position: {:2.1f} M'.format(lane_info["position"]),(10,60), font, 1,(255,255,255),2)
        cv2.putText(result,'confidence: {:3.1f} %'.format(lane_info["confidence"]),(10,90), font, 1,(255,255,255),2)
        return result



if __name__=="__main__":
    from filter import Filter
    from camera import Camera
    import os

    ld = LaneDetector()

    pipeline = Pipeline()
    cam = Camera()
    cam.load_calibration()


    img = cv2.imread(os.path.join("test_images",'test4.jpg'))
    img = cam.undistort(img)
    binary = pipeline.process(img)
    binary_warp,M,inv_M = cam.warp(binary)


    img2 = cv2.imread(os.path.join("test_images",'test1.jpg'))
    img2 = cam.undistort(img2)
    binary2 = pipeline.process(img2)
    binary_warp2,M,inv_M = cam.warp(binary2)

    lane_info = ld.process(binary_warp)
    print(lane_info)
    result = ld.draw(img,lane_info,inv_M)
    cv2.imshow("result",result)
    cv2.waitKey(0)

    lane_info = ld.process(binary_warp2)
    print(lane_info)
    result = ld.draw(img2,lane_info,inv_M)
    cv2.imshow("result",result)
    cv2.waitKey(0)
