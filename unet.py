import keras
import os
from keras import Input
from keras.engine import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, Convolution2D, merge, UpSampling2D, \
    concatenate
from dataset import height, width
from preprocess_images import generate_bw_image
from dataset import generate_patches
from scipy import misc
import numpy as np


def predict_image(image, model, isz):
    imh, imw = image.shape
    image = image[..., None]
    x_len = imw // (isz)
    y_len = imh // (isz)
    print(x_len, y_len)
    resulting_image = np.zeros((image.shape[0], image.shape[1], 3))
    for i in range(y_len):
        for j in range(x_len):
            img_patch = image[i * isz:(i + 1) * isz, j * isz:(j + 1) * isz, :]
            resulting_image[i * isz:(i + 1) * isz, j * isz:(j + 1) * isz] = model.predict(img_patch[None, ...])
    return resulting_image

def get_unet(isz):
    name = "UNET"
    inputs = Input((isz, isz, 1))
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
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00001), loss=keras.losses.mean_absolute_error,
                  metrics=["accuracy", keras.losses.MSE])
    return model, name


def main():
    isz = 512 


    load_weights = True
    weights_file = 'weights/U-NET-DSTL2.hdf5'

    main_model, model_name = get_unet(isz)

    if load_weights == True and os.path.isfile(weights_file):
        print("Loaded weights")
        main_model.load_weights(weights_file)


    y_large = misc.imread("data/Manga546/color.png")

    x_large = generate_bw_image(y_large)

    print("Loaded dataset")
    x, y = generate_patches(x_large, y_large, amount=1500, patch_wh=isz)
    print("Generated patches")
    del x_large, y_large

    checkpointer = ModelCheckpoint('weights/'+model_name+'.hdf5', save_best_only=True)
    main_model.fit(x, y, epochs=24, batch_size=2, validation_split=0.025, callbacks=[checkpointer])

    files = ['660.jpg', '661.jpg', '662.jpg', '663.jpg', '666.jpg', '667.jpg', '668.jpg', '669.jpg']

    for i, filename in enumerate(files):
        misc.imsave("out/"+filename,predict_image(generate_bw_image(misc.imread("data/Manga546/color/"+filename)), main_model, isz))
        


if __name__ == "__main__":
    main()
