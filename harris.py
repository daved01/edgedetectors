'''
Applies Harris corner detector

Last modified: Dec 27, 2020
Author: Dave

Also see: https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html
'''

import numpy as np
import cv2 as cv

img = cv.imread('images/pascal2007-plane.jpg')

# For performance we use a greyscale image
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
grey = np.float32(grey)

# Apply Harris operator
corners = cv.cornerHarris(grey, blockSize=2, ksize=3, k=0.04)

# Colour selected pixels where corners have been detected in red 
img[corners>0.01*corners.max()] = [0,0,255]

# Save image
cv.imwrite("images/harris-example.jpg", img)

# Show image with keypoints
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
