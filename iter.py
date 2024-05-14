import numpy as np
import time
from rgb_not_linear import RGBPicNotLinear
from copy import deepcopy
import cv2


def A(rgb_matrix, hx, hy):
    res = []
    for _ in range(len(rgb_matrix) // hy):
        res.append([])
        for _ in range(len(rgb_matrix[0]) // hx):
            res[-1].append([0, 0, 0])
    for y in range(len(res)):
        for x in range(len(res[0])):
            pixels = [rgb_matrix[yy][xx] for yy in range(hy*y, hy*y + hy) for xx in range(hx*x, hx*x + hx)]
            R = sum([pixel[0] for pixel in pixels]) / len(pixels)
            G = sum([pixel[1] for pixel in pixels]) / len(pixels)
            B = sum([pixel[2] for pixel in pixels]) / len(pixels)
            res[y][x] = [R, G, B]
    return res



def Az_minus_u(rgb_matrix, hx, hy, u):
    Az = A(rgb_matrix, hx, hy)
    # print("Az ", len(Az), len(Az[0]))
    for y in range(len(Az)):
        for x in range(len(Az[0])):
            R, G, B = Az[y][x]
            Ru, Gu, Bu = u[y][x]
            Az[y][x] = [R - Ru, G - Gu, B - Bu]
    return Az


def z_plus_UAz_minus_u(z, uaz):
    temp = deepcopy(z)
    for y in range(len(temp)):
        for x in range(len(temp[0])):
            R, G, B = temp[y][x]
            Ruaz, Guaz, Buaz = uaz[y][x]
            temp[y][x] = [R + Ruaz, G + Guaz, B + Buaz]
    return temp


n_iters = 100

left_dxs = None
left_dys = None
right_neighbour = RGBPicNotLinear("rgb_images/0001.jpg")
right_dxs, right_dys, _ = right_neighbour.process_image("")
for i in range(1, 40):
    print(i)
    start_time = time.time()  # Запускаем таймер
    current_dxs = right_dxs
    current_dys = right_dys
    current = right_neighbour
    if i < 39:
        right_neighbour = RGBPicNotLinear(f'rgb_images/00{("0" + str(i+1)) if (i+1) < 10 else i + 1}.jpg')
        right_dxs, right_dys, _ = right_neighbour.process_image("")
    else:
        right_neighbour = None
        right_dxs, right_dys = None, None

    # print("here2")
    extra_dxs = []
    extra_dys = []
    if left_dxs:
        extra_dxs.append(left_dxs)
    if left_dys:
        extra_dys.append(left_dys)
    if right_dxs:
        extra_dxs.append(right_dxs)
    if right_dys:
        extra_dys.append(right_dys)
    u = current.matrix

    _, _, zk = current.process_image('', extra_dxs=extra_dxs, extra_dys=extra_dys)

    # print("here3")
    for j in range(n_iters):
        print(f'iteration {j}')
        # print("written")
        # print("zk", len(zk), len(zk[0]))
        azkminu = Az_minus_u(zk, 2, 2, u)
        # print("azkminu", len(azkminu), len(azkminu[0]))
        current.matrix = azkminu
        # print(azkminu[:20])
        _, _, uaz = current.process_image("")
        # print(uaz[:20])
        # print("uaz", len(uaz), len(uaz[0]))
        zk1 = z_plus_UAz_minus_u(zk, uaz)
        # print("zk1", len(zk1), len(zk1[0]))
        # zk = zk1
    cv2.imwrite(f'iterative/image_{i}.jpg', np.array(zk))
    end_time = time.time()  # Останавливаем таймер
    execution_time = end_time - start_time
    print("Время выполнения алгоритма для одного кадра:", execution_time, "секунд")
    left_dxs = current_dxs
    left_dys = current_dys
