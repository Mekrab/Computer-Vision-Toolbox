def get_min_and_cords(quards,img_partA_ssd):
    send_back = []
    minElement = np.amin(img_partA_ssd)
    minindex = np.argmin(img_partA_ssd)
    send_back.append(minElement)
    send_back.append(quards[minindex])
    return send_back

