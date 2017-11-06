import sys

import cv2
import numpy as np
from scipy import misc

from dataset import generate_hint
from preprocess_images import generate_adaptive_bw_image
from unet_online_hint import get_unet
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    im_height = 128
    im_width = 128
    main_model, model_name = get_unet(im_width, im_height)

    weights_file = "weights/UNET-ONLINE-HINTED_bug_fix_normal_hint.hdf5"

    main_model.load_weights(weights_file)

    color_img = misc.imread(sys.argv[1])

    plt.imshow(color_img)
    plt.show()
    bw_img = misc.imresize(generate_adaptive_bw_image(color_img),
                           (im_height, im_width))[..., None]
    color_img = misc.imresize(color_img, (im_height, im_width))
    plt.imshow(bw_img[:, :, 0], 'gray')
    plt.show()

    for i in range(0, 10):
        if i == 0:
            hint = cv2.blur(np.concatenate((bw_img, bw_img, bw_img), axis=2),
                            (40, 40))
        elif i == 1:
            hint = np.ones(color_img.shape)*255
        else:
            # hint = cv2.GaussianBlur(color_img, (0, 0), 40)
            hint = cv2.blur(color_img, (i * 10, i * 10))
        plt.imshow(hint)
        plt.show()
        spliced = np.concatenate((bw_img, hint), axis=2) / 255

        prediction = main_model.predict(spliced[None, ...])[0]
        prediction[prediction > 1] = 1
        prediction[prediction < 0] = 0
        plt.imshow(prediction)
        plt.show()


if __name__ == "__main__":
    main()
