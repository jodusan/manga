import sys

from scipy import misc

from preprocess_images import generate_adaptive_bw_image
from unet_online import get_unet


def main():
    im_height = 256
    im_width = 256

    main_model, model_name = get_unet(im_width, im_height)

    weights_file = "weights/UNET-ONLINE.hdf5"

    main_model.load_weights(weights_file)

    color_image = misc.imresize(misc.imread(sys.argv[1]), (im_width, im_height))

    if len(color_image.shape) > 2:
        color_image = generate_adaptive_bw_image(color_image)

    color_image = color_image[None, ..., None] / 255

    misc.imsave("prediction.jpg", main_model.predict(color_image)[0])


if __name__ == "__main__":
    main()
