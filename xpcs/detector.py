#! /usr/bin/env python

import numpy as np

class Detector:
    def __init__(self, name, shape, pixle_size):
        self.name = name
        self.shape = shape
        self.pixle_size = pixle_size

    def qvectors(self, sdd, center, wavelen):
        nrow = self.shape[0]
        ncol = self.shape[1]
        qvec = np.zeros((nrow*ncol, 3), dtype=np.float32)
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
        k0 = 2 * np.pi / wavelen

        # q-vector
        qvec[:,0] = k0 * (cos_al * cos_th - 1).ravel()
        qvec[:,1] = k0 * (cos_al * sin_th).ravel()
        qvec[:,2] = k0 * (sin_al).ravel()
        return qvec

    def qvalues(self, sdd, center, wavelen):
        q = self.qvectors(sdd, center, wavelen)
        return np.linalg.norm(q, axis=1)


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
