### Erosion

erosion = cv2.erode(img,kernel,iterations = 1)

### Dilation

dilation = cv2.dilate(img,kernel,iterations = 1)

### Opening

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

### Closing

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


### Morphological Gradient

gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

### Top Hat

tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

### Black Hat
 
 blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
 
###  Structuring Element
 
cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)

 Elliptical Kernel
 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)

 Cross-shaped Kernel
 cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
