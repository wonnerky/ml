import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K

if __name__ == '__main__':
    # the name of the layer we want to visualize
    # (see model definition at keras/applications/vgg16.py)
    LAYER_NAME = 'block3_conv1'

    # build the VGG16 network with ImageNet weights
    vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    vgg.summary()
    print()
    output_layer = vgg.get_layer(LAYER_NAME).output
    print(output_layer)
    print(output_layer[4,4,4,4])
    print(vgg.input)