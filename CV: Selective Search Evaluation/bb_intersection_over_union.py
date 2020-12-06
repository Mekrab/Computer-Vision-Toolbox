# Def iou for intersection
def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    return_crit = []
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    if 0 < iou <= 0.25:  # Checkcritical regions that never overlap with any other region by more than 25%
        return_crit.append(iou)
        return_crit.append(boxA)
        return_crit.append(boxB)
        # print(return_crit)
        return return_crit

    # Def iou query to grab rois and push to main


def run_iou_query(labels):
    # loop over the labels for each of detected objects in the image
    for label in labels.keys():
        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
    return boxes


temp = []  # Set temp var to grab output from bb_intersection_over_union
crit_iou = []  # Set var to run in program outside of nested FOR
boxes_iou = run_iou_query(labelsF)
for i in range(len(boxes_iou)):
    for j in range(len(boxes_iou)):
        temp = bb_intersection_over_union(boxes_iou[i], boxes_iou[j])
        crit_iou.append(temp)

Not_none_values = filter(None.__ne__, crit_iou)
crit_iou = list(Not_none_values)

print(len(crit_iou))
type(crit_iou)