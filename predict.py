import sys

import cv2
import numpy as np
from matplotlib.pyplot import imshow, show
from scipy import misc

from dataset import generate_hint
from preprocess_images import generate_adaptive_bw_image
from unet_online_hint import get_unet


def main():
    im_height = 256
    im_width = 256
    main_model, model_name = get_unet(im_width, im_height)

    weights_file = "weights/UNET-ONLINE-HINTED.hdf5"

    main_model.load_weights(weights_file)

    color_img = misc.imresize(misc.imread(sys.argv[1]), (256, 256))[:, :, :3]

    imshow(color_img)
    show()
    bw_img = generate_adaptive_bw_image(color_img)[..., None]

    # hint = cv2.GaussianBlur(color_img, (0, 0), 10)
    # hint = hint * 0.3 + np.ones_like(hint) * 0.5 * 255
    # imshow(np.uint8(hint))
    # show()
    #
    # spliced = np.concatenate((bw_img, hint), axis=2) / 255
    #
    # prediction = main_model.predict(spliced[None, ...])[0] * 255
    #
    # imshow(np.uint8(prediction))
    # show()

    hint = cv2.GaussianBlur(color_img, (0, 0), 40)
    imshow(np.uint8(hint))
    show()

    spliced = np.concatenate((bw_img, hint), axis=2) / 255

    prediction = main_model.predict(spliced[None, ...])[0] * 255

    imshow(np.uint8(prediction))
    show()


if __name__ == "__main__":
    main()
