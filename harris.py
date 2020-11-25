'''
Applies Harris corner derector

Last modified: Nov 24, 2020
Author: Dave

Also see: https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html
'''

import numpy as np
import cv2 as cv

img = cv.imread('images/pascal2007-plane.jpg')

# For performance we use a gray scale image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create a 
ddepth = -1
dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

img[dst>0.01*dst.max()] = [0,0,255]

# Save image
cv.imwrite("Harris-example.jpg", img)

# Show image with keypoints
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
