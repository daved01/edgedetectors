'''
Applies Sobel filter

Last modified: Dec 27, 2020
Author: Dave

'''

import numpy as np
import cv2 as cv

# Example image from the Pascal 2007 dataset
img = cv.imread('images/pascal2007-plane.jpg')

# For performance we use a grey scale image
grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
grey = np.float32(grey)

# Apply Sobel operator
ddepth = -1
imgOut = cv.Sobel(grey, ddepth, 0, 1) # Gradients in y direction

# Save image
cv.imwrite("images/sobel-example.jpg", imgOut)

# Show image with gradients
cv.imshow('image', imgOut)
cv.waitKey(0)
cv.destroyAllWindows()
