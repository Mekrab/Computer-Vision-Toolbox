#### Convolutional Neural Networks
- Base Neural Network Applied to Images
- MNIST Digit Classification are displayed in this section of the Computer-Vision-Toolbox but can be seen throughout my work.

Process of CNN at a glance:

- Process: an image as large matrix and a kernel or convolutional matrix as a small matrix that is used for blurring, sharpening, edge detection, and other processing functions. The small kernel sits on top of the big image and slides from left-to-right and top-to-bottom, applying a mathematical operation (i.e., a convolution) at each (x, y)-coordinate of the original image.Hand-define kernels: blurring (average smoothing, Gaussian smoothing, median smoothing, etc.), edge detection (Laplacian, Sobel, Scharr, Prewitt, etc.), and sharpening 

  1. Each layer in a CNN applies a different set of filters, typically hundreds or thousands of them, and combines the results, feeding the output into the next layer in the network.
  - In the context of image classification, our CNN may learn to:
    - Detect edges from raw pixel data in the first layer.
    - Use these edges to detect shapes (i.e., “blobs”) in the second layer.
    - Use these shapes to detect higher-level features such as facial structures, parts of a car, etc.
    - The last layer in a CNN uses these higher-level features to make predictions regarding the contents of the image.
    
  



