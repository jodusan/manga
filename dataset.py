from scipy import misc
import os
import numpy as np
import random

height = 950
width = 640


def load_all_images(folder, bw, resize_x=640, resize_y=950):
    image_list = []

    for filename in sorted(os.listdir(folder)):
        img = misc.imread(os.path.join(folder, filename), bw).astype(np.uint8)
        image_list.append(misc.imresize(img, (resize_y, resize_x)))

    return image_list


def make_giant_image(image_list):
    n = int(np.sqrt(len(image_list)))

    assert np.sqrt(len(image_list)).is_integer(), "Number of images must be a square"

    if len(image_list[0].shape) > 2:
        mat = np.array(image_list).reshape((n, n, height, width, image_list[0].shape[2]))
        result = (mat.swapaxes(1, 2).reshape((height * n, width * n, image_list[0].shape[2])))
    else:
        mat = np.array(image_list).reshape((n, n, height, width))
        result = (mat.swapaxes(1, 2).reshape((height * n, width * n)))

    print(result.shape)
    return result


def generate_patches(image_x, image_y, amount=1000, patch_wh=128):
    result_x = np.empty((amount, patch_wh, patch_wh, 1))
    result_y = np.empty((amount, patch_wh, patch_wh, 3))

    im_height = image_x.shape[0]
    im_width = image_x.shape[1]
    for i in range(amount):
        x = random.randint(0, im_height - patch_wh)
        y = random.randint(0, im_width - patch_wh)
        result_x[i] = image_x[x:x + patch_wh, y:y + patch_wh][..., None]
        result_y[i] = image_y[x:x + patch_wh, y:y + patch_wh]

    return result_x/255, result_y/255


if __name__ == "__main__":
    photo_format = 'png'
    number_of_images = 500

    photo_type = 'color'
    x = load_all_images('train_dataset/' + photo_type, False)
    print("Loaded", len(x), "images")

    r = make_giant_image(x[:number_of_images])
    misc.imsave('train_dataset/' + photo_type + "." + photo_format, r)

    photo_type = 'bw'
    x = load_all_images('train_dataset/' + photo_type, False)
    print("Loaded", len(x), "images")

    r = make_giant_image(x[:number_of_images])
    misc.imsave('train_dataset/' + photo_type + "." + photo_format, r)
