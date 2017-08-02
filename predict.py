import sys

import numpy as np
from scipy import misc

from dataset import generate_hint
from preprocess_images import generate_adaptive_bw_image
from unet_online import get_unet


def main():
    im_height = 256
    im_width = 256
    main_model, model_name = get_unet(im_width, im_height)

    weights_file = "weights/UNET-ONLINE-HINTED.hdf5"

    main_model.load_weights(weights_file)

    color_img = misc.imresize(misc.imread(sys.argv[1]), (256, 256))[:, :, :     3]

    bw_img = generate_adaptive_bw_image(color_img)[..., None]

    hint = generate_hint(color_img)

    spliced = np.concatenate((bw_img, hint), axis=2) / 255

    prediction = main_model.predict(spliced[None, ...])[0]

    misc.imsave("prediction.jpg", prediction)


if __name__ == "__main__":
    main()
