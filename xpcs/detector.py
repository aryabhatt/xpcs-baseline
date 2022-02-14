#! /usr/bin/env python

import numpy as np
from numpy import sin, cos

class Detector:
    def __init__(self, name, shape, pixle_size):
        self.name = name
        self.shape = shape
        self.pixle_size = pixle_size

    def angles(self, sdd, center):
        nrow = self.shape[0]
        ncol = self.shape[1]
        y, x = np.mgrid[0:nrow, 0:ncol]

        # shift coordinate with center
        x = (x - center[1]) * self.pixle_size[0]
        y = (y - center[0]) * self.pixle_size[1]

        # angles 
        # theta
        tmp = np.sqrt(x**2 + sdd**2)
        theta = np.arcsin(x/tmp).ravel()

        # alpha
        tmp2 = np.sqrt(y**2 + tmp**2)
        alpha = np.arcsin(y/tmp2).ravel()

        return theta, alpha

    def qvectors(self, sdd, center, wavelen):
        theta, alpha = self.angles(sdd, center)

        qvec = np.zeros((theta.shape[0], 3), dtype=np.float32)
        # radius of Ewald's sphere
        k0 = 2 * np.pi / wavelen

        # q-vector
        qvec[:,0] = k0 * (cos(alpha) * cos(theta) - 1)
        qvec[:,1] = k0 * (cos(alpha) * sin(theta))
        qvec[:,2] = k0 * (sin(alpha))
        return qvec

    def qvalues(self, sdd, center, wavelen):
        q = self.qvectors(sdd, center, wavelen)
        return np.linalg.norm(q, axis=1)


    def dwba_qvectors(self, sdd, center, wavelen, alphai):
        theta, alpha = self.angles(sdd, center)

        dims = (theta.shape[0], 3)
        # radius of the Ewald's sphere
        k0 = 2 * np.pi / wavelen

        # q-vector
        qx = k0 * (cos(alpha) * cos(theta) - cos(alphai))
        qy = k0 * (cos(alpha) * sin(theta))
        kzf = k0 * sin(alpha)
        kzi = k0 * sin(alphai)

        # (Ti, Tf)
        q1 = np.zeros(dims, dtype=np.float32)
        q1[:,0] = qx
        q1[:,1] = qy
        q1[:,2] = kzf+kzi

        # (Ri, Tf)
        q2 = np.zeros(dims, dtype=np.float32)
        q2[:,0] = qx
        q2[:,1] = qy
        q2[:,2] = kzf-kzi

        # (Ti, Rf)
        q3 = np.zeros(dims, dtype=np.float32)
        q3[:,0] = qx
        q3[:,1] = qy
        q3[:,2] = -kzf+kzi

        # (Ri, Rf)
        q4 = np.zeros(dims, dtype=np.float32)
        q4[:,0] = qx
        q4[:,1] = qy
        q4[:,2] = -kzf-kzi

        return [q1, q2, q3, q4]
        
       

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
