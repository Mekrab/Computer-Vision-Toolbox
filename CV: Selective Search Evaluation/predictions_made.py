labels = {} # empty list for main computation
# Def sequence as a separate function to be used after predictions are made
def pred_made(preds, confidence, local):
    labelFilters = ['beagle']
    labels = {}
    # loop over the predictions
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]
        #print (label) # DEBUG
        if labelFilters is not None and label not in labelFilters:
            continue
        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if prob >= confidence:
            # grab the bounding box associated with the prediction and
            # convert the coordinates
            box = local[i]
            # grab the list of predictions for the label and add the
            # bounding box and probability to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L
    return labels

labels = pred_made(preds, args['conf'], locs)
#print(labels) # DEBUG