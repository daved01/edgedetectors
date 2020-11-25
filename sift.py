'''
Applies SIFT keypoint detector to an input image and displays the keypoints.

Last update: Nov 24, 2020
Author: Dave

Parameters values from paper:
numberOfOctaves = 4
numberOfScaleLevels = 5
sigma = 1.6
k = sqrt(2)
contrastThreshold = 0.03 | Threshold value similar to Harris filter
edgeThreshold = 10
'''

import numpy as np
import cv2 as cv

img = cv.imread('images/pascal2007-plane.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv.xfeatures2d_SIFT.create(nfeatures=300, nOctaveLayers=5, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)
kp = sift.detect(gray, None)

# Draw keypoints
img = cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Save imagee
cv.imwrite('sift-example.jpg', img)

# Show image with keypoints
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
