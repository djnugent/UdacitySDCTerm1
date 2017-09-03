from filter import Filter
from camera import Camera
from lane_detector import LaneDetector
from moviepy.editor import VideoFileClip
import cv2

cam = Camera()
binary_filter = Filter()
lane_detector = LaneDetector()

cam.load_calibration()

def pipeline(img):
    # Preprocess the image before we try to detect lanes
    rectified = cam.undistort(img)
    binary = binary_filter.process(img)
    # Change the perspective to top down
    binary_warped, M, inv_M = cam.perspective_transform(binary)
    # detect lanes in warped filterd image
    lane_info = lane_detector.process(binary_warped)
    # draw the result on original image
    result = lane_detector.draw(img,lane_info,inv_M)
    return result

# test image(save)
import os
img = cv2.imread(os.path.join("test_images",'test4.jpg'))
result = pipeline(img)
cv2.imwrite("output_images/pipeline.png",result)

# Test video(write to file)
'''
output = 'project_video_output2.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(pipeline)
white_clip.write_videofile(output, audio=False)
'''
# Test video(render)
vid = cv2.VideoCapture("project_video.mp4")

while True:
    ret,img = vid.read()
    if not ret:
        break
    result = pipeline(img)
    cv2.imshow("result",result)
    cv2.waitKey(1)
