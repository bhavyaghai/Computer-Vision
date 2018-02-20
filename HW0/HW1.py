# Submitted by:-  Bhavya Ghai
# ID:  111168954

# <h2><center>CV Assignemnt 1</center></h2>
# <p>I have performed addition, subtraction, multiplication, division, resize operations using python 2.7 & opencv3</p>
# This program accepts an input file named "input.jpg"

import cv2
import numpy as np
print("OpenCV version: ", cv2.__version__)

# Load an color image in grayscale
img = cv2.imread('input.jpg', 1)


# Show image on screen
def show(img, label="Input Image"):
    cv2.imshow(label, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# show imput image
show(img)
print("rows, col, channels ", img.shape)
row, col, cha = img.shape
print("Type ", img.dtype)

# Add scalar to image
blank_image = np.zeros((row, col, cha), np.uint8)
blank_image[:] = (100, 100, 100)
img_add = cv2.add(img, blank_image)
show(img_add, "Output: Add by 100")

# Subtract scalar from image
blank_image[:] = (100, 100, 100)
img_sub = cv2.subtract(img, blank_image)
show(img_sub, "Output: Subtract by 100")

# Multiply scalar to image
blank_image[:] = (2, 2, 2)
img_mul = cv2.multiply(img, blank_image)
show(img_mul, "Output: Multiply by 2")

# Divide scalar to image
blank_image[:] = (2, 2, 2)
img_div = cv2.divide(img, blank_image)
show(img_div, "Output: Divide by 2")

# Resize image
img_res = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
show(img_res, "Resized image by 1/2")
