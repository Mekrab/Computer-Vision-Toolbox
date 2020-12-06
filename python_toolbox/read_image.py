
def read_image_files(image_path, conversion=cv2.COLOR_BGR2RGB, image_extensions = ['.JPG', '.JPEG', '.BMP', '.PNG']):
    image_files = glob.glob(os.path.join(image_path, '*'))
    image_files = [item for item in image_files if any([ext in item.upper() for ext in image_extensions])]
    image_list = [(os.path.basename(f),cv2.imread(f, conversion)) for f in image_files]
    image_dict = {file:image for (file,image) in image_list}
    return image_dict