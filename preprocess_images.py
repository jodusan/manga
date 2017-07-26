import numpy as np
from scipy import misc


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

def generate_bw_image(im, axis=2):
    # Alternate version:
    # r = im[:, :, 0] > 50
    # g = im[:, :, 1] > 30
    # b = im[:, :, 2] > 50
    # n = (r * g * b) > 0
    # nn = (r * g * b) == 0
    # im[n] = 255
    # im[nn] = 0
    # return im

    return np.mean(im, axis=axis) > 60


if __name__ == "__main__":
    im = misc.imread("train_dataset/montage_color.jpg")
    generate_bw_image(im)
