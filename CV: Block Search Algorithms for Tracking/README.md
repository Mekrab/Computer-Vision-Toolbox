### The block search in its essencae is used to track images between Image Frames

Multiple uses of tracking  - Stereoscopic - distance/depth estimation, Motion estimation Video stabilization

For searches that do not require a special block:

Full Search:
Multistage Search : Three Step Search (3SS): 
Three Step Search (3SS):
Four Step Search (4SS):

def get_scale(image, posistion, size):
    return image[posistion[0]-size:posistion[0]+size+1,posistion[1]-size:posistion[1]+size+1]
    
Diamond Search (DS):

def block_to_diamond(img):
    img = [[img[0][2]],
           [img[1][1],img[1][3]],
           [img[2][0],img[2][2],img[2][4]],
           [img[3][1],img[3][3]],
           [img[4][2]]]
    return img

Hexagon Block Search (HEXBS):

def block_to_hex(img):
    img = [[img[0][1],img[0][3]],
           [img[2][0],img[2][2],img[2][4]],
           [img[4][1],img[4][3]]]
    return img

#### All these search functions can be put into a ssd function: 

def calculate_ssd(img1, img2):
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)

