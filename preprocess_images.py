import cv2
import numpy as np


def generate_bw_image(img, axis=2):
    return np.mean(img, axis=axis) > 60


def generate_adaptive_bw_image(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adaptive = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
    threshold = gray_image > 40
    bw_image = adaptive * threshold
    return bw_image


def bin_colors(img, number_of_bins=2):
    bin_size = 256 // number_of_bins
    res = np.zeros(img.shape, dtype=np.uint8)
    for i in range(number_of_bins):
        mask = np.logical_and((img >= (bin_size * i)), (img < (bin_size * (i + 1))))
        res += (mask * (bin_size * i + bin_size // 2)).astype(np.uint8)
    return res


def image_segmentation(img, number_of_bins):
    res = np.zeros((img.shape[0], img.shape[1], number_of_bins ** 3), dtype=np.bool)
    bin_size = 256 // number_of_bins
    color_indexed_img = img // bin_size
    for i in range(number_of_bins):
        for j in range(number_of_bins):
            for k in range(number_of_bins):
                r = color_indexed_img[:, :, 0] == i
                g = color_indexed_img[:, :, 1] == j
                b = color_indexed_img[:, :, 2] == k
                mask = r * b * g
                res[:, :, k + j * number_of_bins + i * number_of_bins ** 2] = mask
    return res


def image_desegmentation(input_image, number_of_bins):
    argmax = np.argmax(input_image, axis=2)
    result = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.uint8)
    bin_size = 256 // number_of_bins
    for i in range(number_of_bins ** 3):
        b = i % number_of_bins
        g = i // number_of_bins % number_of_bins
        r = i // (number_of_bins ** 2) % number_of_bins
        pixel_value = bin_size * np.array([r, g, b], dtype=np.uint8) + bin_size // 2
        result[argmax == i] = pixel_value

    return result


if __name__ == "__main__":
    pass
    # bins = 2
    # depth = bins ** 3
    # images = load_all_images('train_dataset/color', False, resize_x=512, resize_y=768)
    # print("loaded dataset")
    # start_time = time.time()
    # y = np.empty((len(images), images[0].shape[0], images[0].shape[1], depth), dtype=np.uint8)
    # for i in range(len(images)):
    #     im = bin_colors(images[i], 2)
    #     y[i] = image_segmentation(im, 2)
    # print(time.time() - start_time)
    # print("finished")
    # start_time = time.time()
    # np.save("train_dataset/color_seg", y)
    # print(time.time() - start_time)
    # del y
    # start_time = time.time()
    # y = np.load("train_dataset/color_seg.npy")
    # print(time.time() - start_time)
