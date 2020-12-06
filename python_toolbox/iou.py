from matplotlib.ticker import MaxNLocator
import itertools


def iou(A, B):
    rectA = list(itertools.chain(*A))  # will need to import itertools
    rectB = list(itertools.chain(*B))

    # Find the overlap coordinates
    xA = max(rectA[0], rectB[0])
    yA = max(rectA[1], rectB[1])
    xB = min(rectA[2], rectB[2])
    yB = min(rectA[3], rectB[3])
    # compute the area of intersection rectangle
    overlap_Area = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the rectangles
    rectA_Area = (rectA[2] - rectA[0]) * (rectA[3] - rectA[1])
    rectB_Area = (rectB[2] - rectB[0]) * (rectB[3] - rectB[1])
    iou = overlap_Area / float(rectA_Area + rectB_Area - overlap_Area)
    return iou

ground_truth = [(3,6),(10,10)]
prediction = [(7,-1),(15,7)]


score = iou(ground_truth, prediction)
print(score)