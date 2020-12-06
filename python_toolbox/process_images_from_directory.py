def process_images_from_directory(faceCascade, directory, scaleFactor=1.1, minNeighbors=8):
    imageDictionary = dict()
    for directoryPath, directoryNames, fileNames in os.walk(directory):                                       # walk path
        for fileName in fileNames:
            imageFile = os.path.join(directoryPath, fileName)                                                 # Join path
            img = cv2.imread(imageFile)                                                                       # Reads image into RGB for histogram comparison
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                    # RGB option (not used)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # GRAY option (used)
            # Scale using faceCascade.detectMultiScale - haarcascade_frontalface_default.xml
            faces = faceCascade.detectMultiScale(img_gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(101,101), maxSize=(800,800))
            # Get rect format for later use
            for (x, y, w, h) in faces:
                image = cv2.rectangle(img_gray,(x,y),(x+w,y+h),(255,255,255),10)                              # NOTE: Putting into white space for easy manipulation
                roi_color = image[y:y + h, x:x + w]                                                           # ROI for better manipulation within program calls
                imageDictionary[fileName] = (img_gray, roi_color, img, imageFile)                             # Push all needed vars into dictionary
    return imageDictionary