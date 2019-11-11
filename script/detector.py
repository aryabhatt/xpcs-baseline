#! /usr/bin/env python

import numpy as np

class Detector:
    def __init__(self, name, shape, pixle_size):
        self.name = name
        self.shape = shape
        self.pixle_size = pixle_size

    def qvectors(self, sdd, center, energy):
        nrow = self.shape[0]
        ncol = self.shape[1]
        y, x = np.mgrid[0:nrow, 0:ncol]

        # shift coordinate with center
        x = (x - center[1]) * self.pixle_size[0]
        y = (y - center[0]) * self.pixle_size[1]

        # angles 
        # theta
        tmp = np.sqrt(x**2 + sdd**2)
        cos_th = sdd / tmp
        sin_th = x / tmp

        # alpha
        tmp2 = np.sqrt(y**2 + tmp**2)
        cos_al = tmp / tmp2
        sin_al = y / tmp2

        # radius of the Ewald's sphere
        k0 = 2 * np.pi * energy / 1.23984

        # q-vector
        qx = k0 * (cos_al * cos_th - 1)
        qy = k0 * (cos_al * sin_th)
        qz = k0 * (sin_al)
        return qx, qy, qz

    def qvalues(self, sdd, center, energy):
        qx, qy, qz = self.qvectors(sdd, center, energy)
        return np.sqrt(qx**2 + qy**2 + qz**2)


class Lambda750k(Detector):
    def __init__(self):
        self.name = 'Lambda 750k'
        self.shape = (512, 1536)
        self.pixle_size = (55.E-06, 55.E-06)

class Square512(Detector):
    def __init__(self):
        self.name = '512x512'
        self.shape = (512, 512)
        self.pixle_size = (100.E-06, 100.E-06)
