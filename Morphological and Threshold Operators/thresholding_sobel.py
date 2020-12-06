import cv2
import numpy as np
img = cv2.imread('dog.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#sobel
sobelX = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
sobelY = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
img_sobel = sobelX + sobelY
#prewitt
kernelX = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernelY = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittX = cv2.filter2D(img, -1, kernelX)
prewittY = cv2.filter2D(img, -1, kernelY)
cv2.imshow("Original Image", img)
cv2.imshow("dog_Sobel X", sobelX)
cv2.imshow("dog_Sobel Y", sobelY)
cv2.imshow("dog_Sobel", img_sobel)
cv2.imshow("dog_Prewitt X", prewittX)
cv2.imshow("dog_Prewitt Y", prewittY)
cv2.imshow("dog_Prewitt", prewittX + prewittY)
cv2.imwrite("dog_Sobel X.jpeg", sobelX)
cv2.imwrite("dog_Sobel Y.jpeg", sobelY)
cv2.imwrite("dog_Sobel.jpeg", img_sobel)
cv2.imwrite("dog_Prewitt_X.jpeg", prewittX)
cv2.imwrite("dog_Prewitt_Y.jpeg", prewittY)
cv2.imwrite("dog_Prewitt.jpeg", prewittX + prewittY)
cv2.waitKey(0)
cv2.destroyAllWindows()

#thresholding
import cv2
import numpy as np
from matplotlib import pyplot as plt
#reading image in gray scale
img = cv2.imread("flower.jpg",0)
cv2.imshow('original image', img)
h,w = np.shape(img)
t0=127
t=127 #initial condition
g1 = []
g2 = []
camel_Prewitt camel_Prewitt_X camel_Prewitt_Y
#calculating t by algorithm
while(1):
for px in range(0,h):
for py in range(0,w):
if (img[px][py] < t):
g1.append(img[px][py])
else:
g2.append(img[px][py])
mu1 = sum(g1) / len(g1)
mu2 = sum(g2) / len(g2)
t0=t
t = ((mu1+ mu2)/2)
delta_t = abs(t-t0)
print(mu1,mu2,t,delta_t)
if(delta_t < 1):
break
print(t)
#calculating the histogram
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
#apply thresholding
for px in range(0,h):
for py in range(0,w):
if (img[px][py] < t):
img[px][py] = 0
else:
img[px][py] = 225
#plotting histogram
plt.plot(hist_full)
plt.xlim([0,256])
plt.ylim(bottom=0)
cv2.imshow('threshold', img)
plt.show()