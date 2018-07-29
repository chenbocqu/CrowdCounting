import numpy as np
import math


class CommonTool:
    def __init__(self):
        """初始化Tool"""
        print("call CommonTool...")

    def get_gaussian_map(self, im, points):

        im_density = np.empty(im.shape())
        print(im_density.shape)
        w, h = im_density.shape

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
            # H=
            x = min(w, max(1, abs(int(math.floor(points[j, 0])))))
            y = min(h, max(1, abs(int(math.floor(points[j, 1])))))

            if x > w | y > h:
                continue

            x1 = x - int(math.floor(f_sz / 2))
            y1 = y - int(math.floor(f_sz / 2))

            x2 = x + int(math.floor(f_sz / 2))
            y1 = y + int(math.floor(f_sz / 2))

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
                # H = fspecial('Gaussian', [double(y2h - y1h + 1), double(x2h - x1h + 1)], sigma)
                H = np.empty(im.shape)

            im_density[y1: y2, x1: x2] = im_density[y1: y2, x1: x2] + H

        return im_density

    def __repr__(self):
        return "This is my CommonTool!"
