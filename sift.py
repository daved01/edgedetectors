'''
Applies SIFT keypoint detector to an input image and displays the keypoints.

Last update: Dec 31, 2020
Author: Dave

Parameters values from paper:
numberOfOctaves = 3
numberOfScaleLevels = 5
sigma = 1.6
k = sqrt(2)
contrastThreshold = 0.09 | Threshold value similar to Harris filter
edgeThreshold = 10
'''

import numpy as np
import cv2 as cv

img = cv.imread('images/pascal2007-plane.jpg')
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv.xfeatures2d_SIFT.create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.09, edgeThreshold=10, sigma=1.6)

# Detect and draw keypoints
kp = sift.detect(grey, None)
img = cv.drawKeypoints(grey,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save images
cv.imwrite('images/sift-example.jpg', img)

# Show image with keypoints
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
