def block_to_cross(img):
    img = [[img[0][0],img[0][2],img[0][4]],
           [img[2][0],img[2][2],img[2][4]],
           [img[4][0],img[4][2],img[4][4]]]
    return img

def block_to_hex(img):
    img = [[img[0][1],img[0][3]],
           [img[2][0],img[2][2],img[2][4]],
           [img[4][1],img[4][3]]]
    return img

def block_to_diamond(img):
    img = [[img[0][2]],
           [img[1][1],img[1][3]],
           [img[2][0],img[2][2],img[2][4]],
           [img[3][1],img[3][3]],
           [img[4][2]]]
    return img