import numpy as np
import itertools
import tiling

import cv2 as cv

class Painter(object):
    def __init__(self, lattice):
        self.n = lattice[0].size
        self.size = 1000 / (2 * self.n)
        self.size = self.size if self.size % 2 == 0 else self.size - 1
        self.xsize = int((self.size * np.sqrt(3) / 2))
        self.img = self.create_image(lattice)

    def create_image(self, lattice):
        x = 1000/2
        y = 1000 - self.size
        img = 255*np.ones((1000, 1000, 3), np.uint8)
        img = self.initialize(x, y, img)
        for i in range(self.n - 1, -1, -1):
            for j in range(self.n - 1, -1, -1):
                img = self.draw_column(x - self.xsize * (i - j), y - self.size / 2 * (i + j - 1), lattice[i][j], img)
        return img

    def draw_lozenge1(self, x, y, img):
        lozenge_x_points = [x, x - int(self.size * np.sqrt(3) / 2), x - int(self.size * np.sqrt(3) / 2), x]
        lozenge_y_points = [y, y + int(self.size / 2), y + int(1.5 * self.size), y + self.size]
        points = np.column_stack((lozenge_x_points, lozenge_y_points)).astype(np.int32)
        img = cv.fillConvexPoly(img, points, (255, 0, 0))
        img = cv.polylines(img, [points], True, 0)
        return img

    def draw_lozenge2(self, x, y, img):
        lozenge_x_points = [x, x + int(self.size * np.sqrt(3) / 2), x + int(self.size * np.sqrt(3) / 2), x]
        lozenge_y_points = [y, y + int(self.size / 2), y + int(1.5 * self.size), y + self.size]
        points = np.column_stack((lozenge_x_points, lozenge_y_points)).astype(np.int32)
        img = cv.fillConvexPoly(img, points, (0, 0, 255))
        img = cv.polylines(img, [points], True, 0)
        return img

    def draw_lozenge3(self, x, y, img):
        lozenge_x_points = [x, x - int(self.size * np.sqrt(3) / 2), x, x + int(self.size * np.sqrt(3) / 2)]
        lozenge_y_points = [y, y + int(self.size / 2), y + self.size, y + int(0.5 * self.size)]
        points = np.column_stack((lozenge_x_points, lozenge_y_points)).astype(np.int32)
        img = cv.fillConvexPoly(img, points, (0, 255, 255))
        img = cv.polylines(img, [points], True, 0)
        return img

    def initialize(self, x, y, img):
        for i, j in itertools.product(range(self.n), range(self.n)):
            self.draw_lozenge3(x - self.xsize*(i - j), y - self.size/2 * (i + j), img)
            self.draw_lozenge1(x - self.xsize*i, y - self.size*self.n - self.size*j + self.size/2 * i, img)
            self.draw_lozenge2(x + self.xsize * i, y - self.size * self.n - self.size * j + self.size / 2 * i, img)
        return img

    def draw_column(self, x, y, height, img):
        height = int(height)
        for i in range(1, height + 1):
            img = self.draw_lozenge1(x + self.xsize, y - i * self.size, img)
            img = self.draw_lozenge2(x - self.xsize, y - i * self.size, img)
        return self.draw_lozenge3(x, y - height * self.size - self.size/2, img)

def show_image(lattice):
    p = Painter(lattice)
    cv.imshow('lattice', p.img)
    cv.waitKey(0)

def save_image(lattice, filename):
    p = Painter(lattice)
    cv.imwrite(filename, p.img)

tiling = tiling.Tiling(20, 100000)
tiling.metropolis(100000, thermalization=10000)
print('done')
save_image(tiling.to_3d_lattice(np.round(tiling.average_configuration).astype(np.int32)), "lattice.jpg")
show_image(tiling.to_3d_lattice(np.round(tiling.average_configuration).astype(np.int32)))

