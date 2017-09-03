import matplotlib.pyplot as plt
import numpy as np
import cv2
from lesson_functions import *
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
import imageio

#gif = imageio.get_writer("../output_images/sliding_window.gif",fps = 20)


class CarDetector():

    def __init__(self,search_levels,horizon_pix=450, center_pix = 640, orient=9,pix_per_cell=8,cell_per_block=2,spatial_size=(16,16),hist_bins=32):

        self.svc =  joblib.load( open("svm.pkl", "rb" ) )
        self.X_scaler =  joblib.load( open("scaler.pkl", "rb" ) )

        # Feature Extractor parameters
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins

        # Most cars appear on horizon
        self.horizon_pix = horizon_pix
        self.center_pix = center_pix

        # Level is defined by (window size, number of windows, % overlap)
        self.search_levels = search_levels

        self.bbox_queque = []
        self.rolling_heatmap = np.zeros((720,1280))

        self.calculate_windows()


    # calculate_windows - Calculates sliding windows bounding boxes based on a criteria
    def calculate_windows(self):
        # Extract image info
        x_center = int(self.center_pix)

        self.windows = [] # (upperleft_x, upperleft_y, width, height)
        # Calculate windows for each search level
        for window_center, window_size, window_num,overlap in self.search_levels:
            total_width = (1-overlap) * (window_num - 1) * window_size + window_size
            left_start = int(x_center - total_width/2)
            upperleft_y = int(window_center - window_size/2)
            width = window_size
            height = window_size
            for i in range(0,window_num):
                upperleft_x = int(left_start + (i * (1-overlap) * window_size))
                self.windows.append((upperleft_x,upperleft_y,width,height))

    # detect - run detection pipeline on image
    def detect(self,img,draw = False):

        # Using sliding window to find cars
        bbox_list = self.find_cars_raw(img)

        self.bbox_queque = [bbox_list]# + self.bbox_queque[:29]

        filtered_cars, heat, heatmap = self.heat_filter(img,self.bbox_queque)


        if draw:
            raw = self.draw(bbox_list)
            filtered = self.draw(filtered_cars)
            return filtered_cars, raw, filtered, heat, heatmap

        return filtered_cars

    # find_cars_raw - find cars using sliding window technique
    def find_cars_raw(self,image):

        self.y_start = 200
        self.y_stop = 1000

        # Crop image around largest search Level
        img = image[self.y_start:self.y_stop,:,:]

        # Change the color space
        img = convert_color(img, cspace='HLS')

        bbox_list = []
        for window in self.windows:
            x,y,w,h = window

            #cv2.rectangle(image,(x, y),(x+w, y+h),(0,0,255),3)
            #gif.append_data(image)
            #continue

            # Extract the image patch
            subimg = cv2.resize(img[y-self.y_start:y-self.y_start+h, x:x+w], (64,64))

            # Extract Features
            feature = [extract_feature(subimg, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)]

            # Scale features and make a prediction
            feature = self.X_scaler.transform(feature)
            prediction = self.svc.predict(feature)

            if prediction == 1:
                bbox_list.append(((x, y),(x+w,y+h)))
                #cv2.rectangle(draw_img,(x, y_start + y),(x+w,y_start + y+h),(0,0,255),3)

        #gif.close()
        return bbox_list

    # heat_filter - filter raw detections using a heatmap
    def heat_filter(self,img,bbox_list):
        threshold = 3

        # flatten list
        bbox_list = [item for sublist in bbox_list for item in sublist]

        # Create a heatmap
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heat[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        self.rolling_heatmap = self.rolling_heatmap * 0.8 + heat * 0.2

        # threshold heat to remove false positives
        heatmap[self.rolling_heatmap > threshold] = 1
        #heatmap = self.rolling_heatmap

        # Visualize the heatmap when displaying
        heatmap = np.clip(heatmap, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)

        filtered_bbox = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            filtered_bbox.append(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

        return filtered_bbox,heat, self.rolling_heatmap

    # draw - draws a bounding box list onto an image
    def draw(self,bbox_list):
        draw_img = np.copy(img)
        for bbox in bbox_list:
            cv2.rectangle(draw_img,bbox[0],bbox[1],(0,0,255),3)
        return draw_img

if __name__=="__main__":

    # define our search levels (horizon pixel, window size, number of windows, overlap %)
    search_levels = []
    search_levels.append((440,70,86,0.8))
    search_levels.append((460,120,49,0.8))
    search_levels.append((470,160,36,0.8))
    search_levels.append((475,180,31,0.8))
    search_levels.append((480,240,22,0.8))

    # Initialize a car detector
    cd = CarDetector(search_levels = search_levels,\
                    horizon_pix = 450, \
                    center_pix = 640, \
                    orient = 9,\
                    pix_per_cell = 8, \
                    cell_per_block = 2,\
                    spatial_size = (16,16), \
                    hist_bins = 32)



    '''
    # Open video
    vid = cv2.VideoCapture("../project_video.mp4")
    writer = imageio.get_writer('../project_video_output3.mp4', fps=30)

    while True:
        # Read frame
        ret,img = vid.read()
        if not ret:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run pipeline
        bbox_list,raw,filtered,heat,heatmap = cd.detect(img,draw=True)

        # Stitch into one image for viewing
        final = np.zeros_like(raw)
        raw = cv2.resize(raw,(640,360))
        filtered = cv2.resize(filtered,(640,360))
        heat = cv2.resize(heat,(640,360))
        heatmap = cv2.resize(heatmap,(640,360))
        final[:360,:640,:] = raw
        final[:360,640:,:] = filtered
        heat = heat * 8
        final[360:,:640,:] = np.dstack(((heat,heat,heat)))
        heatmap *= 20
        final[360:,640:,:] = np.dstack(((heatmap,heatmap,heatmap)))

        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(final,'Single Frame Detection',(260,15), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(final,'Final Detection',(940,15), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(final,'Single Frame Heatmap',(250,375), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(final,'Expo Moving Average Heatmap',(880,375), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        writer.append_data(final)

        #cv2.imshow("result",final)
        #cv2.waitKey(1)


    writer.close()
    '''


    '''
    # New Sliding window result
    img = imread('../test_images/test1.jpg')
    bbox_list,raw,filtered,heat,heatmap = cd.detect(img,draw=True)
    fig = plt.figure()
    plt.imshow(raw)
    fig.tight_layout()
    fig.savefig("../output_images/new_result1.png",bbox_inches='tight')
    plt.show()
    '''


    # Heatmap filter
    fig = plt.figure(figsize=(8,12))

    for i in range(1,7):
        img = imread('../test_images/test{}.jpg'.format(i))
        bbox_list,raw,filtered,heat,heatmap = cd.detect(img,draw=True)
        plt.subplot(6,2,2*i-1)
        plt.imshow(raw)
        #plt.title('Car Positions')
        plt.subplot(6,2,2*i)
        plt.imshow(heat, cmap='hot')
        #plt.title('Heat Map')
    fig.tight_layout()
    fig.savefig("../output_images/heatmap.png",bbox_inches='tight')
    plt.show()
