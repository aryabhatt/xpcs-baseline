#! /usr/bin/env python

import mdscatter
import math
import numpy as np
import os
import re
import h5py
import time
import matplotlib.pyplot as plt

from loader import list_lammps_txt_files, load_lammps_txt
from fresnel import propagation_coeffs
from detector import Square512 as Detector

if __name__ == '__main__':


    det = Detector()
    energy = 10000
    wavelen = 1.23984E+04/energy
    sdd = 2
    alphai = (math.pi / 180.) * 0.2
    center = [0, 256]
    outfile = 'gixpcs.h5'
    refidx = 4.88E-06 + 1j * 7.37E-08

    # move md locations to different origin and scale
    origin = [8, 8, 8]
    scale = 28

    # input data
    datadir = os.path.join(os.getcwd(), '../lammps')
    filename_pattern = 'al*.txt'
    txtfiles = list_lammps_txt_files(datadir, filename_pattern)
    Nsteps = len(txtfiles)
    print('Number of steps', Nsteps)
    

    # output hdf5 file
    h5f = h5py.File(outfile, 'w')
    grp = h5f.create_group('gixpcs')
    qtmp = grp.create_dataset('q_points', (3, *det.shape), 'f')
    dset = grp.create_dataset('imgs', (Nsteps, *det.shape), 'f')    

    # undistorted q-vectors
    qvecs = det.dwba_qvectors(sdd, center, wavelen, alphai)

    # dwba coefficients
    thata, alpha = det.angles(sdd, center)
    fc = propagation_coeffs(alpha, alphai, refidx)
    
    # turn the crank
    t0 = time.time()
    Nsteps = 2
    for i in range(Nsteps):
        mdsim = load_lammps_txt(txtfiles[i], origin=origin, scale=scale)
        pts = mdsim['POSITIONS']
        ff = np.zeros_like(thata, dtype=np.complex_)
        for i in range(4):
            ff += fc[i] * mdscatter.dft(pts, qvecs[i])
        img = np.abs(ff)**2
        img = np.reshape(img, det.shape)
        plt.imshow(np.log(img+1), origin='lower')
        plt.show()
        dset[i,:,:] = np.reshape(img, det.shape)
    t1 = time.time() - t0
    print('time taken = %f\n' % t1)
    h5f.close()