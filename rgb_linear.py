from copy import deepcopy
from tools import *
import cv2
import numpy as np
import time

class RGBPicLinear:

    def __init__(self, path, K=K1, hx=2, hy=2):
        self.matrix = cv2.imread(path, cv2.IMREAD_COLOR)
        self.Ny = len(self.matrix)
        self.Nx = len(self.matrix[0])
        self.K = K
        self.hx = hx
        self.hy = hy

    def print_matrix(self):
        for y in range(self.Ny):
            print(*self.matrix[y])

    def process_image(self, name, extra_fjs=None, write_mode=False):
        # расширяется изначальная матрица
        res = []
        for y in range(self.Ny):
            for _ in range(self.hy):
                res.append(self.matrix[y])
                for x in range(self.Nx):
                    for _ in range(self.hx - 1):
                        ls = list(res[-1])
                        # print(ls)
                        ls.insert(x * self.hx, res[-1][x * self.hx])
                        res[-1] = np.array(ls)
        # print(res[0][0])
        fj = []
        for y in range(self.Ny*self.hy):
            fj.append([])
            for x in range(self.Nx*self.hx):
                fj[-1].append([sum([res[y][x][j] * self.K(x/self.hx - i) for i in range(self.Nx)]) for j in range(3)])
        fj_to_return = deepcopy(fj)
        if extra_fjs:
            for extra_fj in extra_fjs:
                for y in range(self.Ny * self.hy):
                    for x in range(self.Nx * self.hx):
                        for i in range(3):
                            fj[y][x][i] += extra_fj[y][x][i]
            for y in range(self.Ny * self.hy):
                for x in range(self.Nx * self.hx):
                    for i in range(3):
                        fj[y][x][i] /= len(extra_fjs) + 1
        for y in range(self.Ny * self.hy):
            for x in range(self.Nx * self.hx):
                for i in range(3):
                    res[y][x][i] = sum([fj[y][x][i] * self.K(y/self.hy - j) for j in range(len(fj))])
        if write_mode:
            cv2.imwrite(name, np.array(res))
        return fj_to_return


left_fj = None
right_neighbour = RGBPicLinear("rgb_images/0001.jpg")
right_fj = right_neighbour.process_image("_")
for i in range(1, 4):
    print(i)
    start_time = time.time()  # Запускаем таймер
    current_fj = right_fj
    current = right_neighbour
    if i < 3:
        right_neighbour = RGBPicLinear(f'rgb_images/00{("0" + str(i + 1)) if (i + 1) < 10 else i + 1}.jpg')
        right_fj = right_neighbour.process_image("")
    else:
        right_neighbour = None
        right_fj = None
    extra_fjs = []
    if left_fj:
        extra_fjs.append(left_fj)
    if right_fj:
        extra_fjs.append(right_fj)
    current.process_image(f'rgb_linear/image_{i}.jpg', extra_fjs=extra_fjs, write_mode=True)
    end_time = time.time()  # Останавливаем таймер
    execution_time = end_time - start_time
    print("Время выполнения алгоритма для одного кадра:", execution_time, "секунд")
    left_fj = current_fj

