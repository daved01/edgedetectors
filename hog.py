'''
Applies the steps of HOG to an image, preferrably of size 64x128. Visualizes the intermediate steps as well as the final feature map.

Last modified: Nov 25, 2020
Author: Dave

'''

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import hog as hogsk
from skimage import data, exposure

# Read image
imgorg = cv.imread('images/pascal2007-person-correctsize.jpg') #  Use the correct size of 64x128 pixels
img = np.float32(imgorg) / 255.0

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Calculate gradient
gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)

# Find magnitude and orientation of gradients
mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

# Plot x, y, and magnitude of gradients
plt.subplot(1,3,1)
plt.imshow(gx, cmap='Blues')
plt.title('Gradient x')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(gy, cmap='Blues')
plt.title('Gradient y')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,3)
plt.imshow(mag, cmap='Blues')
plt.title('Gradient magnitude')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('hog-example-gradients.png')
plt.show()

# Make final plot
implot = cv.cvtColor(imgorg, cv.COLOR_BGR2RGB)

## Calculate HOG. Opencv doesn't offer visualization so skimage is used here.
fd, hogimage = hogsk(implot, visualize=True)
hogimage = exposure.rescale_intensity(hogimage, in_range=(0, 10))

## Make grid 8x8 pixels for visualization
dx = 8
dy = 8
colour = [255, 255, 255]

implot[:,::dx,:] = colour
implot[::dy,:,:] = colour
mag[:,::dx,:] = colour
mag[::dy,:,:] = colour

plt.subplot(1,3,1)
plt.imshow(implot)
plt.title('Original')
plt.xticks([])
plt.yticks([])
plt.subplot(1,3,2)
plt.imshow(mag, cmap='Blues')
plt.title('Gradient magnitude')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(hogimage, cmap='Blues')
plt.title('HOG feature map')
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig('hog-example.png')
plt.show()