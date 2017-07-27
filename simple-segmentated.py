import keras
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Activation, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization
from dataset import height, width
from preprocess_images import generate_bw_image
from dataset import generate_patches
from scipy import misc
import numpy as np


def predict_image(image, model, isz):
    imh, imw = image.shape
    image = image[..., None]
    x_len = imw // (isz )
    y_len = imh // (isz )
    print(x_len, y_len)
    resulting_image = np.ones((image.shape[0], image.shape[1], 3))
    for i in range(y_len):
        for j in range(x_len):
            img_patch = image[i * isz:(i + 1) * isz, j * isz:(j + 1) * isz, :]
            #print(img_patch.shape)
            #print((i * isz, (i + 1) * isz, j * isz, (j + 1) * isz))
            prediction = model.predict(img_patch[None, ...])
            #prediction = misc.imresize(prediction, (isz, isz, 3))
            resulting_image[i * isz:(i + 1) * isz, j * isz:(j + 1) * isz] = prediction 
    return resulting_image


if __name__ == "__main__":
    isz = 128 

    test_model = Sequential()
    test_model.add(Conv2D(10, kernel_size=(3, 3), padding='same', input_shape=(isz, isz, 1)))
    test_model.add(BatchNormalization())
    test_model.add(Conv2D(32, kernel_size=(3, 3), padding='same'))
    test_model.add(BatchNormalization())
    test_model.add(Conv2D(3, kernel_size=(6,6), padding='same', activation='relu'))
    #test_model.add(Conv2D(3, (2, 2), activation='relu', padding='same'))
    test_model.compile(loss=keras.losses.MSE,
                       optimizer=keras.optimizers.Adam(lr=0.0001),
                       metrics=['accuracy'])

    y_large = misc.imread("train_dataset/color.png")

    x_large = generate_bw_image(y_large)

    print("Loaded dataset")
    x, y = generate_patches(x_large, y_large, amount=5000, patch_wh=isz)
    

    print("Generated patches")
    del x_large, y_large

    test_model.fit(x, y, epochs=10, batch_size=32, validation_split=0.1)

    # inpt = x[900:]
    # res = test_model.predict(inpt)

    misc.imsave("fajl.png", predict_image(generate_bw_image(misc.imread("train_dataset/color/1.jpg")), test_model, isz))
