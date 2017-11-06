import os

import keras
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Conv2D, Dropout, MaxPooling2D, UpSampling2D, \
    concatenate

from dataset import image_loader_generator


def get_selu_unet(isx, isy):
    name = "UNET-ONLINE-HINTED-SELU_point_generator"
    inputs = Input((isx, isy, 4))
    conv1 = Conv2D(32, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(inputs)
    conv1 = Conv2D(32, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(pool1)
    conv2 = Conv2D(64, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(pool2)
    conv3 = Conv2D(128, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(pool3)
    conv4 = Conv2D(256, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], 3)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(256, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(up6)
    conv6 = Conv2D(256, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], 3)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(128, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(up7)
    conv7 = Conv2D(128, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], 3)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(64, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(up8)
    conv8 = Conv2D(64, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], 3)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(32, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(up9)
    conv9 = Conv2D(32, (3, 3), activation="selu", padding="same",
                   kernel_initializer="lecun_normal")(conv9)

    conv10 = Conv2D(3, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00005),
                  loss=keras.losses.MSE,
                  metrics=["accuracy", keras.losses.MSE])
    return model, name


def get_unet(isx, isy):
    name = "UNET-ONLINE-HINTED_bug_fix_normal_hint"
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

    conv10 = Conv2D(3, (1, 1), activation="linear")(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00005),
                  loss=keras.losses.MSE,
                  metrics=["accuracy", keras.losses.MSE])
    return model, name


def main():
    im_height = 128
    im_width = 128

    load_weights = True

    main_model, model_name = get_unet(im_width, im_height)

    weights_file = "weights/" + model_name + '.hdf5'

    if load_weights and os.path.isfile(weights_file):
        print("Loaded weights")
        main_model.load_weights(weights_file)

    training_images = 51168
    val_images = 10080
    training_batch_size = 32
    val_batch_size = 2

    checkpointer = ModelCheckpoint('weights/' + model_name + '.hdf5',
                                   save_best_only=True)

    tensorboard = keras.callbacks.TensorBoard(
        log_dir='logs/' + model_name + '/',
        histogram_freq=0,
        write_graph=True,
        write_images=False)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                  factor=0.8,
                                                  patience=3,
                                                  )

    train_generator = image_loader_generator("datasets/manga_dataset_train/",
                                             False,
                                             resize_x=im_width,
                                             resize_y=im_height,
                                             batch_size=training_batch_size,
                                             generate_bw=True)

    val_generator = image_loader_generator("datasets/manga_dataset_val/",
                                           False,
                                           resize_x=im_width,
                                           resize_y=im_height,
                                           batch_size=val_batch_size,
                                           generate_bw=True)

    main_model.fit_generator(train_generator,
                             training_images // training_batch_size,
                             epochs=10,
                             verbose=1,
                             callbacks=[checkpointer, tensorboard, reduce_lr],
                             validation_data=val_generator,
                             validation_steps=val_images // val_batch_size)


if __name__ == "__main__":
    main()
