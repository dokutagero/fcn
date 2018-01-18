import numpy as np


def transform_lsvrc2012_vgg16(inputs):
    img = inputs[0]

    # LSVRC2012 used by VGG16
    MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= MEAN_BGR
    img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W

    transformed = list(inputs)
    transformed[0] = img
    return transformed

def transform_bridge_vgg16(inputs):
    img = inputs[0]

    # LSVRC2012 used by VGG16

    #without calculating in the pixels that are non_deck
    MEAN_BGR = np.array([115.263115, 125.08375,  129.77165 ]) 
    #mean over whole image, including non-deck
    #MEAN_BGR = np.array([121.254555, 129.5794,   137.16048 ]) 



    img = img[:, :, ::-1]  # RGB -> BGR
    img = img.astype(np.float32)
    img -= MEAN_BGR
    img = img.transpose(2, 0, 1)  # H, W, C -> C, H, W

    transformed = list(inputs)
    transformed[0] = img
    return transformed
