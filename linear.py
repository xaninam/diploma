from copy import deepcopy
from tools import *
import cv2
import numpy as np


class WBPicLinear:

    def __init__(self, path, K=K1, hx=2, hy=2):
        self.matrix = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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
                        res[-1] = np.insert(res[-1], x*self.hx, res[-1][x*self.hx])

        fj = []
        for y in range(self.Ny*self.hy):
            fj.append([])
            for x in range(self.Nx*self.hx):
                fj[-1].append(sum([res[y][x] * self.K(x/self.hx - i) for i in range(self.Nx)]))
        fj_to_return = deepcopy(fj)
        if extra_fjs:
            for extra_fj in extra_fjs:
                for y in range(self.Ny * self.hy):
                    for x in range(self.Nx * self.hx):
                        fj[y][x] += extra_fj[y][x]
            for y in range(self.Ny * self.hy):
                for x in range(self.Nx * self.hx):
                    fj[y][x] /= len(extra_fjs) + 1
        for y in range(self.Ny * self.hy):
            for x in range(self.Nx * self.hx):
                res[y][x] = sum([fj[y][x] * self.K(y/self.hy - j) for j in range(len(fj))])
        if write_mode:
            cv2.imwrite(name, np.array(res))
        return fj_to_return


left_fj = None
right_neighbour = WBPicLinear("images/0505(1)_000.jpg")
right_fj = right_neighbour.process_image("_")
for i in range(31):
    print(i)
    current_fj = right_fj
    current = right_neighbour
    if i < 30:
        right_neighbour = WBPicLinear(f'images/0505(1)_0{("0" + str(i + 1)) if (i + 1) < 10 else i + 1}.jpg')
        right_fj = right_neighbour.process_image("")
    else:
        right_neighbour = None
        right_fj = None
    extra_fjs = []
    if left_fj:
        extra_fjs.append(left_fj)
    if right_fj:
        extra_fjs.append(right_fj)
    current.process_image(f'linear/image_{i}.jpg', extra_fjs=extra_fjs, write_mode=True)
    left_fj = current_fj
