import cv2
import numpy as np

img = cv2.imread('dog.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#sobel
sobelx = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_8U,0,1,ksize=3)
sobel = sobelx + sobely

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewittx = cv2.filter2D(img, -1, kernelx)
prewitty = cv2.filter2D(img, -1, kernely)


cv2.imshow("Original Image", img)

cv2.imshow("Sobel X", sobelx)
cv2.imshow("Sobel Y", sobely)
cv2.imshow("Sobel", sobel)
cv2.imshow("Prewitt X", prewittx)
cv2.imshow("Prewitt Y", prewitty)
cv2.imshow("Prewitt", prewittx + prewitty)


cv2.waitKey(0)
cv2.destroyAllWindows()