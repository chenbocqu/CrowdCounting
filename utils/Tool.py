import cv2
import numpy as np
import math


def gaussian_kernel(kernel_x=15, kernel_y=15, sigma=4.0):
    kx = cv2.getGaussianKernel(kernel_x, sigma)
    ky = cv2.getGaussianKernel(kernel_y, sigma)
    H = np.multiply(kx, np.transpose(ky))
    return np.transpose(H)


def rbg2gray(img):
    weight = [0.2125, 0.7154, 0.0721]
    return np.dot(img, weight)


def density_map(im, points):
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
        H = gaussian_kernel(kernel_x=f_sz, kernel_y=f_sz, sigma=sigma)

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
            kx = y2h - y1h + 1
            ky = x2h - x1h + 1
            H = gaussian_kernel(kernel_x=ky, kernel_y=kx, sigma=sigma)

        im_density[y1: y2 + 1, x1: x2 + 1] = im_density[y1: y2 + 1, x1: x2 + 1] + H

    return im_density
