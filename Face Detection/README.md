#### Face Detection - 


- This program uses Haar-cascade Detection in OpenCV

  1. OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any object like car, planes etc. you can use OpenCV to create one. Its full details are given here: Cascade Classifier Training. This assignment deals with detection not training. OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are stored in opencv/data/haarcascades/ folder. Let’s create face and eye detector with OpenCV. First we need to load the required XML classifiers. Then load our input image (or video) in grayscale mode.



