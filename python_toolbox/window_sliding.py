rois = []
locs = []

start = cv2.getTickFrequency()   # Set clock for run time
# Go to window sliding Region proposal algorithms
def window_sliding(image_loop_yield,WINDOW_STEP, ROI_SIZE):
    # Set local roi's and loacl bounds found to push to main
    rois = []
    locs = []
    for image in image_loop_yield:    # loop over images to yeild dimensions of the next image in the looping methoud
        scale = W / float(image.shape[1])
        for (x, y, roiOrig) in window(image, WINDOW_STEP, ROI_SIZE): # Go to slide window for dividing the input
            x = int(x * scale)                                       #image into a list of roi and respective boxes
            y = int(y * scale)        # Set scaling for window
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            roi = cv2.resize(roiOrig, INPUT_SIZE)  # read roi
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
            rois.append(roi)           # push locals and bounds to main
            locs.append((x, y, x + w, y + h))
    return rois, locs
end = cv2.getTickFrequency()

rois, locs =  window_sliding(image_loop_yield, WINDOW_STEP, ROI_SIZE)