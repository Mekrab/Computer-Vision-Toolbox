def get_shapes(roi):
    rows,cols = roi.shape
    s= []
    for x in range (1,15):
        for y in range (1,15):
            s.append(roi[x-1:x+2,y-1:y+2])
            if len(s)<3:
                pass
    return s


