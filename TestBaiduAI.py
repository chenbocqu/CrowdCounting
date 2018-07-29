import matplotlib.image as mpimg
import numpy as np
import math
import json


# import cv2

def rbg2gray(img):
    weight = [0.2125, 0.7154, 0.0721]
    return np.dot(img, weight)


def gaussian_2d_kernel(kernel_size=3, sigma=0):
    kernel = np.zeros([kernel_size, kernel_size])
    center = kernel_size // 2

    if sigma == 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = 2 * (sigma ** 2)
    sum_val = 0
    for i in range(0, kernel_size):
        for j in range(0, kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / s)
            sum_val += kernel[i, j]
            # /(np.pi * s)
    sum_val = 1 / sum_val
    return kernel * sum_val


def gaussian_kernel(kx=15, ky=15, sigma=0):
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


# def gaussian_kernel(kernel_x=15, kernel_y=15, sigma=4.0):
#     kx = cv2.getGaussianKernel(kernel_x, sigma)
#     ky = cv2.getGaussianKernel(kernel_y, sigma)
#     return np.multiply(kx, np.transpose(ky))


# 自定义方法
def get_gaussian_map(im, points):
    w, h, c = im.shape

    if c == 3:
        im = rbg2gray(im)

    im_density = np.zeros(im.shape)
    print(im_density.shape)

    # 长度为0返回
    if len(points) == 0:
        return im_density

    # 长度为1
    if len(points) == 1:
        x1 = max(1, min(w, round(points[0, 0])))
        y1 = max(1, min(h, round(points[0, 1])))

        im_density[y1, x1] = 255
        return im_density

    for j in range(len(points)):
        f_sz = 15
        sigma = 4.0

        # 获取一个高斯函数矩阵
        H = gaussian_kernel(kernel_size=f_sz, sigma=sigma)

        x = min(w, max(1, abs(int(math.floor(points[j, 0])))))
        y = min(h, max(1, abs(int(math.floor(points[j, 1])))))

        if x > w | y > h:
            continue

        radius = int(math.floor(f_sz / 2))

        x1, x2 = x - radius, x + radius
        y1, y2 = y - radius, y + radius

        dfx1, dfy1, dfx2, dfy2 = 0, 0, 0, 0

        change_H = False

        if x1 < 1:
            dfx1 = abs(x1) + 1
            x1 = 1
            change_H = True

        if y1 < 1:
            dfy1 = abs(y1) + 1
            y1 = 1
            change_H = True

        if x2 > w:
            dfx2 = x2 - w
            x2 = w
            change_H = True

        if y2 > h:
            dfy2 = y2 - h
            y2 = h
            change_H = True

        x1h = 1 + dfx1
        y1h = 1 + dfy1
        x2h = f_sz - dfx2
        y2h = f_sz - dfy2

        if change_H == True:
            print('H is changed !')
            kx = y2h - y1h + 1
            ky = x2h - x1h + 1
            # # H = fspecial('Gaussian', [double(y2h - y1h + 1), double(x2h - x1h + 1)], sigma)

            # im_density[y1: y2 + 1, x1: x2 + 1] = im_density[y1: y2 + 1, x1: x2 + 1] + H

    return im_density


img_fold = "E:/baidu_AI/baidu_star_2018/image/"

# 读取数据
with open('E:/baidu_AI/baidu_star_2018/annotation/annotation_train_stage1.json', 'r') as f:
    ann_train = json.load(f)

ann = ann_train.get('annotations')
length = len(ann)

# 读取图片
for i in range(length):
    print('处理 : %d ' % i)
    ann_i = ann[i]
    cnt = ann_i.get('num')

    I = mpimg.imread(img_fold + ann_i.get('name'))
    locs = ann_i.get('annotation')

    count = 1  # 标记patch
    gt4map = np.array([])

    if i == 3:
        break

    if ann_i.get('type') == 'dot':
        for j in range(len(locs)):
            x = locs[j].get('x')
            y = locs[j].get('y')
            point = np.array([x, y])
            if j == 0:
                gt4map = np.array([x, y])
            else:
                gt4map = np.vstack([gt4map, point])

    if ann_i.get('type') == 'bbox':
        continue

    file_name = ann_i.get('name')

    h, w, c = I.shape
    # create density map
    d_map_h = math.floor(math.floor(h / 2.0) / 2.0)
    d_map_w = math.floor(math.floor(w / 2.0) / 2.0)

    d_map = get_gaussian_map(I, gt4map)

    # d_map = create_density(gt / 4.0, d_map_h, d_map_w)

    p_h = math.floor(I.shape[0] / 3.0)
    p_w = math.floor(I.shape[1] / 3.0)
    d_map_ph = p_h
    d_map_pw = p_w

    # create non-overlapping patches of images and density maps
    py = 0
    py2 = 0
    for j in range(3):
        px = 0
        px2 = 0
        for k in range(3):
            final_image = np.array(I[py: py + p_h - 1, px: px + p_w - 1, :], dtype="double")
            print(I.shape)
            print(final_image.shape)

            # final_gt = d_map[py2: py2 + d_map_ph - 1, px2: px2 + d_map_pw - 1]

            px = px + p_w
            px2 = px2 + d_map_pw

            # if final_image.shape[2] < 3:
            #     final_image = repmat(final_image, [1, 1, 3])

            # image_name = sprintf('%s_%d.jpg', file_name, count)
            # gt_name = sprintf('%s_%d.mat', file_name, count)
            # imwrite(uint8(final_image), fullfile(final_image_fold, image_name))
            # do_save(fullfile(final_gt_fold, gt_name), final_gt)
            count = count + 1
        py = py + p_h
        py2 = py2 + d_map_ph
