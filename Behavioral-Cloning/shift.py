import cv2
import glob
import os

import numpy as np
images = glob.glob("IMG/center*.jpg")



def perspective_transform(img):
    # Choose an offset from image corners to plot detected corners
    offset = 200 # offset for dst points x value
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    # Specify the transform
    shift = -30
    scale = 0
    src = np.float32([[142,76],[190,76],[243,97],[121,91]])
    ul = 152,76
    ur = 197,76
    lr = 262,97
    ll = 141,91

    dst = np.float32([ul,ur,lr,ll])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


def crop(img):
    return img[50:-20]

def color(img,mode):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if mode == 0:
        img = np.dstack((img,np.zeros_like(img),np.zeros_like(img)))
    if mode == 1:
        img = np.dstack((np.zeros_like(img),img,np.zeros_like(img)))
    if mode == 2:
        img = np.dstack((np.zeros_like(img),np.zeros_like(img),img))
    return img

for img_name in images:
    center_name = os.path.join('./IMG/',img_name.split('\\')[-1])
    left_name = center_name.replace("center","left")
    center_img = cv2.imread(center_name)
    left_img = cv2.imread(left_name)



    warp = perspective_transform(center_img)

    center_img = crop(center_img)
    left_img = crop(left_img)
    warp = crop(warp)

    result = cv2.addWeighted(color(center_img,0), 0.5, color(left_img,1), 0.5, 0)

    result2 = cv2.addWeighted(color(left_img,0), 0.5, color(warp,1), 0.5, 0)

    cv2.imshow("center",center_img)
    cv2.imshow("left",left_img)
    cv2.imshow("warp",warp)
    cv2.imshow("result",result)
    cv2.imshow("result2",result2)

    cv2.waitKey(0)
