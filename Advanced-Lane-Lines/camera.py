import pickle
import cv2
import numpy as np
import os
import glob



class Camera():

    def __init__(self):
        self.mtx = None
        self.dist = None


    def load_calibration(self):
        ## Read in the saved camera matrix and distortion coefficients
        try:
            dist_pickle = pickle.load( open('dist_file.p', "rb" ) )
            self.mtx = dist_pickle["mtx"]
            self.dist = dist_pickle["dist"]
            return True
        except IOError:
            return False

    def calibrate(self, internal_corners, image_folder):

        nx,ny = internal_corners

        images = glob.glob(os.path.join(image_folder,'calibration*.jpg'))

        objpoints = []
        imgpoints = []

        objp = np.zeros((nx*ny,3),np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        for fname in images:
            print("Processing " + fname)
            img = cv2.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found, save points
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)

        # Calibrate camera
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape, None, None)

        #save distortion matrix/coef
        pickle.dump( {"mtx":self.mtx,"dist":self.dist}, open( "dist_file.p", "wb" ) )
        print("Saving calibration to dist_file.p")

    def undistort(self,img):
        if self.mtx is None or self.dist is None:
            raise("Please load calibration or run Camera.calibrate()")
        return cv2.undistort(img, self.mtx, self.dist, None, self.mtx)


    def perspective_transform(self,img):
        # Choose an offset from image corners to plot detected corners
        offset = 200 # offset for dst points x value
        # Grab the image shape
        img_size = (img.shape[1], img.shape[0])
        # Specify the transform
        src = np.float32([[150+430,460],[1150-440,460],[1150,720],[150,720]])
        dst = np.float32([[offset, 0],
                          [img_size[0]-offset, 0],
                          [img_size[0]-offset, img_size[1]],
                          [offset, img_size[1]]])

        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        inv_M = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, M, img_size)
        return warped, M, inv_M

if __name__=="__main__":
    import matplotlib.pyplot as plt

    cam = Camera()

    # load calibration or calibrate camera
    if( not cam.load_calibration()):
        print("No calibrate found. Calibrating camera...")
        cam.calibrate((9,6),"camera_cal")
    else:
        print("Calibration loaded")

    # undistort image
    img = cv2.imread(os.path.join("camera_cal",'calibration1.jpg'))
    result = cam.undistort(img)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(result)
    ax2.set_title('Undistorted Result', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig("output_images/distortion_result.png")
    plt.show()

    # Test on an actual image
    img = cv2.imread(os.path.join("test_images",'test2.jpg'))
    result = cam.undistort(img)
    cv2.imwrite("output_images/test_distortion_result.png",img)

    # Warp the image
    img = cv2.imread(os.path.join("test_images",'test1.jpg'))
    birds_eye, M, inv_M = cam.perspective_transform(img)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    f.tight_layout()
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(birds_eye)
    ax2.set_title('Warped Result', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    f.savefig("output_images/warp_result.png")
    plt.show()
