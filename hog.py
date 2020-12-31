'''
Applies the steps of HOG to an image, preferrably of size 64x128. Visualizes the intermediate steps as well as the final feature map.

Last modified: Dec 31, 2020
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

# Calculate gradient
gx = cv.Sobel(img, cv.CV_32F, 1, 0, ksize=1)
gy = cv.Sobel(img, cv.CV_32F, 0, 1, ksize=1)

# Find magnitude and orientation of gradients
mag, angle = cv.cartToPolar(gx, gy, angleInDegrees=True)

# Plot x, y, and magnitude of gradients
titles = ['Gradient x', 'Gradient y', 'Gradient magnitude']
for i in range(0,3):
    print(i)
    plt.subplot(1,3,i+1)
    plt.imshow(gx, cmap='Blues')
    plt.title(titles[i], color='white')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig('images/hog-example-gradients.png', transparent=True)
plt.show()

# Calculate HOG. Opencv doesn't offer visualization so skimage is used here.
implot = cv.cvtColor(imgorg, cv.COLOR_BGR2RGB)
fd, hogimage = hogsk(implot, visualize=True)
hogimage = exposure.rescale_intensity(hogimage, in_range=(0, 10))

# Make grid 8x8 pixels for visualization
dx = 8
dy = 8
colour = [255, 255, 255]

implot[:,::dx,:] = colour
implot[::dy,:,:] = colour
mag[:,::dx,:] = colour
mag[::dy,:,:] = colour

titles = ['Original', 'Gradient magnitude', 'HOG feature map']
images = [implot, mag, hogimage]

for i in range(0,3):
    plt.subplot(1,3,i+1)
    plt.imshow(images[i])
    plt.title(titles[i], color='white')
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.savefig('images/hog-example.png', transparent=True)
plt.show()
