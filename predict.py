import keras
from keras import Input
from keras.engine import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, Convolution2D, merge, UpSampling2D, \
    concatenate
from dataset import height, width, load_all_images, generate_hint
from preprocess_images import generate_bw_image, generate_adaptive_bw_image
from dataset import generate_patches, image_loader_generator
from scipy import misc
import numpy as np
import os
import sys
from unet_online import get_unet 

def main():
    im_height = 256
    im_width = 256
    main_model, model_name = get_unet(im_width, im_height)

    weights_file = "weights/UNET-ONLINE-HINTED.hdf5"

    main_model.load_weights(weights_file)

    color_img = misc.imresize(misc.imread(sys.argv[1]), (256, 256))[:, :, :     3]
    
    bw_img = generate_adaptive_bw_image(color_img)[..., None]

    hint = generate_hint(color_img, wo=True, str=40)
    
    spliced = np.concatenate((bw_img, hint), axis=2) / 255

    prediction = main_model.predict(spliced[None, ...])[0]
    
    misc.imsave("prediction.jpg", prediction)


if __name__ == "__main__":
    main()
