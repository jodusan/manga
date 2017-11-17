import sys

from matplotlib.pyplot import imshow, show, imsave
from scipy import misc

from preprocess_images import generate_adaptive_bw_image
from unet_online import get_unet
import numpy as np
import os


def main():
    im_height = 256
    im_width = 256

    main_model, model_name = get_unet(im_width, im_height)

    weights_file = "weights/UNET-ONLINE.hdf5"

    main_model.load_weights(weights_file)

    files = os.listdir(sys.argv[1])

    for file in files:
        color_image = misc.imresize(
            misc.imread(os.path.join(sys.argv[1], file)), (im_width, im_height))

        imsave("results/" + file + "_original.png", color_image)

        if len(color_image.shape) > 2:
            color_image = generate_adaptive_bw_image(color_image)

        imsave("results/" + file + "_lines.png", color_image, cmap='gray')

        color_image = color_image[None, ..., None] / 255

        imsave("results/" + file + "_predicted.png",
               np.uint8(main_model.predict(color_image)[0] * 255))


if __name__ == "__main__":
    main()
