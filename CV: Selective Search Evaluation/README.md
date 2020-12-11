#### Selective Search for Object Detection

This functions provided in the folder can help adapt object classification for object detection.

Using the following:
- Windowing
- R-CNN
- Fast R-CNN
- Faster R-CNN
- Selective Search

#### Sliding Window
-In this approach, a sliding window is moved over the image, and all the pixels inside that sliding window are cropped out and sent to an image classifier.
-If the image classifier identifies a known object, the bounding box and the class label are stored. Otherwise, the next window is evaluated.

#### R-CNN
- Region proposals are merely lists of bounding boxes with a small probability of containing an object. It did not know or care which object was contained in the bounding box.
- A region proposal algorithm outputs a list of a few hundred bounding boxes at different locations, scales, and aspect ratios. 

#### Selective Search
- Selective Search algorithm takes these over-segments as initial input and performs the following steps:
  1. Add all bounding boxes corresponding to segmented parts to the list of regional proposals
  2. Group adjacent segments based on similarity
  3. Loops back to step # 1
  
#### Fast R-CNN
- Classify the region into one of the classes ( e.g. dog, cat, background ).
- Improve the accuracy of the original bounding box using a bounding box regressor.

#### Faster R-CNN 
- A Convolutional Neural Network is used to produce a feature map of the image which was simultaneously used for training a region proposal network and an image classifier.



