import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math


def generate_gaussian_mask(shape, sigma, sigma_y=None):
    if sigma_y == None:
        sigma_y = sigma
    rows, cols = shape

    def get_gaussian_fct(size, sigma):
        fct_gaus_x = np.linspace(0, size, size)
        fct_gaus_x = fct_gaus_x - size / 2
        fct_gaus_x = fct_gaus_x ** 2
        fct_gaus_x = fct_gaus_x / (2 * sigma ** 2)
        fct_gaus_x = np.exp(-fct_gaus_x)
        return fct_gaus_x

    mask = np.outer(get_gaussian_fct(rows, sigma), get_gaussian_fct(cols, sigma_y))
    return mask


def gaussian_kernel(kx=15, ky=15, sigma=4):
    kernel = np.zeros([kx, ky])
    center_i = kx // 2
    center_j = ky // 2

    if sigma == 0:
        sigma = ((kx - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kx):
        for j in range(0, ky):
            x = i - center_i
            y = j - center_j
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
            # /(np.pi * s)
    sum_val = 1 / sum_val
    return kernel * sum_val


img_fold = "E:/baidu_AI/baidu_star_2018/image/"

# 读取数据
with open('E:/baidu_AI/baidu_star_2018/annotation/annotation_train_stage1.json', 'r') as f:
    ann_train = json.load(f)

ann = ann_train.get('annotations')

# test func
print(generate_gaussian_mask([3, 2], 4))

# 读取图片
for i in [1]:
    print('处理 id : %d ' % i)

    ann_i = ann[i]
    cnt = ann_i.get('num')

    I = mpimg.imread(img_fold + ann[i].get('name'))

    w, h, c = I.shape
    locs = ann_i.get('annotation')

    plt.imshow(I)
    plt.title(ann[i].get('name'))
    plt.show()

    count = 1  # 标记patch
    gt4map = np.array([])

    p_h = math.floor(I.shape[0] / 3.0)
    p_w = math.floor(I.shape[1] / 3.0)
    d_map_ph = p_h
    d_map_pw = p_w

    py = 0
    py2 = 0
    for j in range(3):
        px = 0
        px2 = 0
        for k in range(3):
            start_y, end_y = int(py), int(py + p_h)
            start_x, end_x = int(px), int(px + p_w)

            print(end_y)
            for fi in range(start_y, end_y):
                print(fi)

            final_image = np.array(I[start_y: end_y, start_x: end_x, :], dtype="double")
            # final_gt = np.array(d_map[int(py2): int(py2 + d_map_ph), int(px2): int(px2 + d_map_pw)], dtype="double")

            plt.subplot(121)
            plt.imshow(final_image)
            plt.subplot(122)
            # plt.imshow(final_gt)
            plt.show()

            px = px + p_w + 1
            px2 = px2 + d_map_pw + 1

            # if final_image.shape[2] < 3:
            #     final_image = repmat(final_image, [1, 1, 3])

            # image_name = sprintf('%s_%d.jpg', file_name, count)
            # gt_name = sprintf('%s_%d.mat', file_name, count)
            # imwrite(uint8(final_image), fullfile(final_image_fold, image_name))
            # do_save(fullfile(final_gt_fold, gt_name), final_gt)
            count = count + 1
        py = py + p_h + 1
        py2 = py2 + d_map_ph + 1
