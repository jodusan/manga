from scipy import misc
import os
import numpy as np
import random

from preprocess_images import generate_adaptive_bw_image

height = 950
width = 640


def is_bw(image):
    mask = image[:, :, 0] * (image[:, :, 0] == image[:, :, 1]) == image[:, :, 2]
    if np.sum(mask) >= image.shape[0] * image.shape[1] - 1000:
        return True
    return False


def load_all_images(folder, bw, resize_x=640, resize_y=950):
    image_list = []

    for filename in sorted(os.listdir(folder)):
        img = misc.imread(os.path.join(folder, filename), bw).astype(np.uint8)
        if not bw:
            if not is_bw(img):
                image_list.append(misc.imresize(img, (resize_y, resize_x)))

    return image_list


def image_loader_generator(folder, bw, resize_x=640, resize_y=950, batch_size=1000, generate_bw=False):
    filename_list = sorted(os.listdir(folder))

    i = 0
    while True:
        image_batch_color = []
        if generate_bw:
            image_batch_bw = []
        j = 0
        while j < batch_size and i < len(filename_list):
            img = misc.imread(os.path.join(folder, filename_list[i]), bw).astype(np.uint8)
            if not bw:
                if not is_bw(img):
                    resized_color_image = misc.imresize(img, (resize_y, resize_x))
                    image_batch_color.append(resized_color_image)
                    if generate_bw:
                        image_batch_bw.append(generate_adaptive_bw_image(resized_color_image))
                else:
                    j -= 1
            j += 1
            i = (i + 1) % len(filename_list)
        if generate_bw:
            yield (np.array(image_batch_bw)/255)[..., None], np.array(image_batch_color)/255
        else:
            yield np.array(image_batch_color)/255


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

    return result_x / 255, result_y / 255


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
