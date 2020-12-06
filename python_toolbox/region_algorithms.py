# Def sliding window for Region proposal algorithms
def window(image, stepsize, win_sliding):
    # code sequence for a windowing approach to dividing the input image into a list of roi and respective boxes
    for y in range(0, image.shape[0] - win_sliding[1], stepsize):
        for x in range(0, image.shape[1] - win_sliding[0], stepsize):
            # get current window
            yield (x, y, image[y:y + win_sliding[1], x:x + win_sliding[0]])
# Def image loop for Region proposal algorithms
def image_loop(image, scale=1.5, minSize=(224, 224)):
    # get original image
    yield image
    # keep looping over the image loop
    while True:
        # compute the dimensions of the next image in the looping methoud
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # brek loop if new image does not == supplied minimum size
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # get the next image in the loop
        yield image