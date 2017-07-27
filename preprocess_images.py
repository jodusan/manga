from pprint import pprint

import numpy as np
from scipy import misc
import time

#
# height = 950
# width = 640
# main_dataset = "scott_dataset"
# output_dataset = "train_dataset"
# kernel = np.ones((3, 3), np.uint8)
#
# number = 0
# # Process bw images
# for filename in sorted(os.listdir(main_dataset + "/bw")):
#     for file_i, img_filename in enumerate(sorted(os.listdir(main_dataset + "/bw/" + filename))):
#         print(file_i, filename)
#         img = cv2.imread(os.path.join(main_dataset, "bw", filename, img_filename), 0)
#         # img = cv2.resize(img, (width, height))
#         cv2.imwrite(output_dataset + "/bw/" + str(number) + '.jpg', img)
#         number += 1
#
# # Process color images
# number = 0
#
# for filename in sorted(os.listdir(main_dataset + "/bw")):
#     for file_i, img_filename in enumerate(sorted(os.listdir(main_dataset + "/color/" + filename))):
#         print(file_i, filename)
#         img = cv2.imread(os.path.join(main_dataset, "color", filename, img_filename))
#         # img = cv2.resize(img, (width, height))
#         cv2.imwrite(output_dataset + "/color/" + str(number) + '.jpg', img)
#         number += 1
#
# coorelation_coefs = []
# # Proveravati korelaciju slika preko: sum((255-bw) - (255-bw(color))) ako previse odudara onda izbaciti
# for filename in sorted(os.listdir(output_dataset + "/bw")):
#     bw = cv2.imread(os.path.join(output_dataset, "bw", filename), 0)
#     bw = cv2.resize(bw, (width, height))
#
#     color = cv2.imread(os.path.join(output_dataset, "color", filename))
#     color = cv2.resize(color, (width, height))
#     gray_image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
#     gray_image = cv2.resize(gray_image, (width, height))
#     # 68240079
#     # 80061947
#     # 51651614
#     # print 255-bw
#     # gray_image = (gray_image > 128).astype(np.int32)*255
#     gray_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
#     # gray_image = cv2.Canny(gray_image, 100, 200)
#     # gray_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
#     coorelation_coef = np.sum(np.abs((255 - gray_image) - (255 - bw)))
#     if coorelation_coef > 50000000:
#         print(filename)
#         os.remove(os.path.join(output_dataset, "bw", filename))
#         os.remove(os.path.join(output_dataset, "color", filename))
#     else:
#         cv2.imwrite(os.path.join(output_dataset, "bw", filename), bw)
#         cv2.imwrite(os.path.join(output_dataset, "color", filename), color)
#         coorelation_coefs.append(coorelation_coef)
#         # break
#
# print(np.percentile(coorelation_coefs, 2))
# print(np.percentile(coorelation_coefs, 5))
# print(np.percentile(coorelation_coefs, 95))
# print(np.percentile(coorelation_coefs, 98))
# print(len(coorelation_coefs))
# plt.hist(coorelation_coefs)
# plt.show()
# print(coorelation_coefs)
from dataset import load_all_images


def generate_bw_image(img, axis=2):
    return np.mean(img, axis=axis) > 60


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
    pprint(argmax)
    result = np.zeros((input_image.shape[0], input_image.shape[1], 3), dtype=np.uint8)
    bin_size = 256 // number_of_bins
    for i in range(number_of_bins**3):
        b = i % number_of_bins
        g = i // number_of_bins % number_of_bins
        r = i // (number_of_bins ** 2) % number_of_bins
        pixel_value = bin_size * np.array([r, g, b], dtype=np.uint8) + bin_size // 2
        result[argmax == i] = pixel_value

    return result


if __name__ == "__main__":
    bins = 2
    depth = bins ** 3
    images = load_all_images('train_dataset/color', False, resize_x=512, resize_y=768)
    print("loaded dataset")
    start_time = time.time()
    y = np.empty((len(images), images[0].shape[0], images[0].shape[1], depth), dtype=np.uint8)
    for i in range(len(images)):
        im = bin_colors(images[i], 2)
        y[i] = image_segmentation(im, 2)
    print(time.time() - start_time)
    print("finished")
    start_time = time.time()
    np.save("train_dataset/color_seg", y)
    print(time.time() - start_time)
    del y
    start_time = time.time()
    y = np.load("train_dataset/color_seg.npy")
    print(time.time() - start_time)
