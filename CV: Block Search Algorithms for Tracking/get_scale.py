def get_scale(image, posistion, size):
    return image[posistion[0]-size:posistion[0]+size+1,posistion[1]-size:posistion[1]+size+1]

