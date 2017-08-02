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

def get_unet(isx, isy):
    name = "UNET-ONLINE-HINTED"
    inputs = Input((isx, isy, 4))
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding="same")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], 3)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(up6)
    conv6 = Conv2D(256, (3, 3), activation="relu", padding="same")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], 3)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(up7)
    conv7 = Conv2D(128, (3, 3), activation="relu", padding="same")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], 3)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(up8)
    conv8 = Conv2D(64, (3, 3), activation="relu", padding="same")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], 3)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(up9)
    conv9 = Conv2D(32, (3, 3), activation="relu", padding="same")(conv9)

    conv10 = Conv2D(3, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.000005),
                  loss=keras.losses.MSE,
                  metrics=["accuracy", keras.losses.MSE])
    return model, name


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
