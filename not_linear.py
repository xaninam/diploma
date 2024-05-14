from copy import deepcopy
import cv2
import numpy as np


class WBPicNotLinear:

    def __init__(self, path, hx=2, hy=2):
        self.matrix = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.Ny = len(self.matrix)
        self.Nx = len(self.matrix[0])
        self.hx = hx
        self.hy = hy

    def print_matrix(self):
        for y in range(self.Ny):
            print(*self.matrix[y])

    def process_image(self, name, extra_dxs=None, extra_dys=None, write_mode=False):
        res = []
        dxs = []
        dys = []
        for y in range(self.Ny):
            dxs.append(list(self.matrix[y]))
            dys.append(list(self.matrix[y]))
            for _ in range(self.hy):
                res.append(self.matrix[y])
                for x in range(self.Nx):
                    for _ in range(self.hx - 1):
                        res[-1] = np.insert(res[-1], x*self.hx, res[-1][x*self.hx])
        # cv2.imwrite('ouput1.png', np.array(res))
        # print(self.matrix)
        for y in range(self.Ny):
            for x in range(self.Nx):
                if x != self.Nx - 1:
                    # print(1, self.matrix[y][x + 1], self.matrix[y][x], self.matrix[y][x + 1] - self.matrix[y][x],
                    #       type(self.matrix[y][x]))
                    dx = min(255, abs(int(self.matrix[y][x + 1]) - int(self.matrix[y][x])) + 1)
                    # print(dx)
                    dxs[y][x] = dx
                else:
                    dx = min(255, abs(int(self.matrix[y][x]) - int(self.matrix[y][x - 1])) + 1)
                    dxs[y][x] = dx
                if y != self.Ny - 1:
                    dy = min(255, abs(int(self.matrix[y + 1][x]) - int(self.matrix[y][x])) + 1)
                    dys[y][x] = dy
                else:
                    dy = min(255, abs(int(self.matrix[y][x]) - int(self.matrix[y - 1][x])) + 1)
                    dys[y][x] = dy
        dxs_to_return = deepcopy(dxs)
        dys_to_return = deepcopy(dys)
        if extra_dxs:
            for extra_dx in extra_dxs:
                for y in range(self.Ny):
                    for x in range(self.Nx):
                        dxs[y][x] += extra_dx[y][x]
            for y in range(self.Ny):
                for x in range(self.Nx):
                    dxs[y][x] /= len(extra_dxs) + 1
        if extra_dys:
            for extra_dy in extra_dys:
                for y in range(self.Ny):
                    for x in range(self.Nx):
                        dys[y][x] += extra_dy[y][x]
            for y in range(self.Ny):
                for x in range(self.Nx):
                    dys[y][x] /= len(extra_dys) + 1
        for y in range(self.Ny*self.hy):
            for x in range(self.Nx*self.hx):
                if x % self.hx != 0 or y % self.hy != 0:
                    luc_x = x//self.hx
                    luc_y = y//self.hy
                    if luc_x == self.Nx - 1 and luc_y == self.Ny - 1:
                        pixels = [[luc_x, luc_y], [luc_x, luc_y], [luc_x, luc_y], [luc_x, luc_y]]
                    elif luc_x == self.Nx - 1:
                        pixels = [[luc_x, luc_y], [luc_x, luc_y + 1], [luc_x, luc_y], [luc_x, luc_y + 1]]
                    elif luc_y == self.Ny - 1:
                        pixels = [[luc_x, luc_y], [luc_x + 1, luc_y], [luc_x, luc_y], [luc_x + 1, luc_y]]
                    else:
                        pixels = [[luc_x, luc_y], [luc_x, luc_y + 1], [luc_x + 1, luc_y], [luc_x + 1, luc_y + 1]]
                    distances = []
                    for pixel in pixels:
                        px, py = pixel
                        px *= self.hx
                        py *= self.hy
                        distances.append((abs(px - x)**2 + abs(py - y)**2)**0.5)
                    # print(pixels, distances)
                    res[y][x] = self.weighted_sum(pixels, distances, dxs, dys)
                    # print(res[y][x])
        if write_mode:
            cv2.imwrite(name, np.array(res))
        return dxs_to_return, dys_to_return

    def weighted_sum(self, pixels, distances, dxs, dys):
        weights = [1, 1, 1, 1]
        sd = sum([1/d for d in distances])
        sdxs = 0
        sdys = 0
        for i in range(4):
            x, y = pixels[i]
            sdxs += 1/dxs[y][x]
            sdys += 1/dys[y][x]
        for i in range(4):
            weights[i] *= (1/distances[i])/sd
        sw = sum(weights)
        for i in range(4):
            weights[i] /= sw
        for i in range(4):
            x, y = pixels[i]
            weights[i] *= (1/dxs[y][x])/sdxs
        sw = sum(weights)
        for i in range(4):
            weights[i] /= sw
        for i in range(4):
            x, y = pixels[i]
            weights[i] *= (1/dys[y][x])/sdys
        sw = sum(weights)
        for i in range(4):
            weights[i] /= sw
        res = 0
        for i in range(4):
            x, y = pixels[i]
            weight = weights[i]
            res += self.matrix[y][x] * weight
        return res


left_dxs = None
left_dys = None
right_neighbour = WBPicNotLinear("images/0505(1)_000.jpg")
right_dxs, right_dys = right_neighbour.process_image("")
for i in range(31):
    print(i)
    current_dxs = right_dxs
    current_dys = right_dys
    current = right_neighbour
    if i < 30:
        right_neighbour = WBPicNotLinear(f'images/0505(1)_0{("0" + str(i + 1)) if (i + 1) < 10 else i + 1}.jpg')
        right_dxs, right_dys = right_neighbour.process_image("")
    else:
        right_neighbour = None
        right_dxs, right_dys = None, None
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
    current.process_image(f'not_linear/image_{i}.jpg', extra_dxs=extra_dxs, extra_dys=extra_dys, write_mode=True)
    left_dxs = current_dxs
    left_dys = current_dys
