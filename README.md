# Coin_Detection-



# Name: Sanjay V
# Reg no : 212223230188
# Date : 04/11/25
# Aim:

To develop a Python program using image processing techniques to detect and count coins in an image by applying filtering, edge detection, and contour detection methods.


# Procedure:

1.Import Libraries Import necessary Python libraries:
2.Read the Input Image Load the image containing coins:
3.Preprocess the Image Apply Gaussian Blur to reduce noise:
4.Apply Edge Detection Use Canny Edge Detection to find edges:
5.Find Contours Detect the outlines of coins:
6.Draw Contours and Count Coins Draw contours and count the number of detected coins:
7.Display the Output Show the final image with detected coins:

# Program:

```
# NAME : SANJAY V
# REG NO : 212223230188

import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

image=cv2.imread('CoinsA.png')

imageCopy = image.copy()
plt.imshow(image[:,:,::-1]);
plt.title("Original Image")
plt.show()

imageGray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,12))
plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image")
plt.subplot(122); plt.imshow(imageGray,cmap='gray');plt.title("Grayscale Image");
plt.show()

imageB,imageG,imageR = cv2.split(image)

plt.figure(figsize=(20,12))
plt.subplot(141);plt.imshow(image[:,:,::-1]);plt.title("Original Image")
plt.subplot(142);plt.imshow(imageB,cmap='gray');plt.title("Blue Channel")
plt.subplot(143);plt.imshow(imageG,cmap='gray');plt.title("Green Channel")
plt.subplot(144);plt.imshow(imageR,cmap='gray');plt.title("Red Channel");
plt.show()

thresh = 20
maxValue = 255

th,dst_bin=cv2.threshold(imageG,thresh,maxValue,cv2.THRESH_BINARY_INV)

plt.imshow(dst_bin,cmap='gray');
plt.title("Threshold Binary Inverse");
plt.show()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
imageDilated2 = cv2.dilate(dst_bin, kernel, iterations=2)
dilated_image_rgb = cv2.cvtColor(imageDilated2, cv2.COLOR_BGR2RGB)
plt.imshow(dilated_image_rgb,cmap='gray');plt.title('Dilated Image Iteration 2');plt.show()

kSize=(5,5)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kSize)
eroded_image = cv2.erode(dilated_image_rgb, kernel, iterations=2)
imageEroded = cv2.cvtColor(eroded_image, cv2.COLOR_BGR2RGB)

plt.imshow(imageEroded,cmap='gray');plt.title("Eroded Image");plt.show()

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0

params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(imageEroded)
print(f"Number of coins detected: {len(keypoints)}")

```
# Output :

<img width="397" height="435" alt="image" src="https://github.com/user-attachments/assets/041b47ed-34b8-4620-b74a-0df2a01dc430" />
<img width="986" height="524" alt="image" src="https://github.com/user-attachments/assets/294b763d-349b-4666-b461-31b2b690886a" />
<img width="397" height="435" alt="image" src="https://github.com/user-attachments/assets/024fb2f7-9c46-4107-a772-c465a0c2d010" />
<img width="397" height="435" alt="image" src="https://github.com/user-attachments/assets/61d5c324-c837-4187-8a54-2e906988769a" />
<img width="397" height="435" alt="image" src="https://github.com/user-attachments/assets/ac5cc26a-262b-4ae9-a6a9-296c64a141db" />

# RESULT :
Number of coins detected: 9
