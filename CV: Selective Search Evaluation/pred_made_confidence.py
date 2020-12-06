labels_topconf_F = {}
labels_topconf_Q = {}

labels_topconf_F = {}
labels_topconf_Q = {}

def pred_made_confidence(preds, confidence, boxes):
    labelFilters = ['beagle']
    L = []
    # loop over the predictions
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]
        #print (label)

        # only if the label filters are not empty *and* the label does not
        # exist in the list, then ignore it
        if labelFilters is not None and label not in labelFilters:
            continue
        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if  prob > confidence:
            pass

            L.append((prob))
    return L
# Define 0.7 for bottom parameter, no 100% find was made in testing so .99 was not needed to define
argsF['conf']=0.7
argsQ['conf']=0.7

labels_topconf_F = pred_made_confidence(predsF, argsF['conf'], boxesF)
labels_topconf_Q = pred_made_confidence(predsQ, argsQ['conf'],boxesQ)


plt.figure(figsize=(10,5))
plt.title('Fast Search')
plt.ylabel('Frequency of Occurrence')
plt.xlabel('Confidence Level')
plt.hist(labels_topconf_F, 100, density =True, color = 'lightblue')
plt.show()

plt.figure(figsize=(10,5))
plt.title('Quality Search')
plt.ylabel('Frequency of Occurrence')
plt.xlabel('Confidence Level')
plt.hist(labels_topconf_Q, 200, density =True, color = 'lightblue')
plt.show()

