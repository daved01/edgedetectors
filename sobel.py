'''
Applies Sobel filter

Last modified: Nov 25, 2020
Author: Dave

'''

import numpy as np
import cv2 as cv

img = cv.imread('images/pascal2007-plane.jpg')

# For performance we use a gray scale image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

gray = np.float32(gray)

# Create a 
ddepth = -1
dst = cv.Sobel(gray, ddepth, 1, 0) # Gradients in x direction

img[dst>0.01*dst.max()] = [0,0,255]

# Save image
cv.imwrite("sobel-example.jpg", img)

# Show image with keypoints
cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()
