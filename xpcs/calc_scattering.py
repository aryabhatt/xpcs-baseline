#! /usr/bin/env python

import mdscatter
import numpy as np
import os
import re
import h5py
import time
import matplotlib.pyplot as plt
from loader import list_lammps_txt_files, load_lammps_txt
from detector import Lambda750k, Square512

if __name__ == '__main__':

    wavelen = 0.1127
    energy = 1.23984 / wavelen
    
    sdd = 4.
    scale = 28
    center = (0, 768)

    detector = Lambda750k()
    qvecs = detector.qvectors(sdd, center, wavelen)

    outf = 'xpcs_out.h5'
    h5f = h5py.File(outf, 'w')
    grp = h5f.create_group('xpcs')
    qtmp = grp.create_dataset('q_points', (3, *detector.shape), 'f')

    # read data
    datadir = '../lammps'
    pattern = 'al.*.txt'
    txtfiles = list_lammps_txt_files(datadir, pattern)
    Nsteps = len(txtfiles)
    dset = grp.create_dataset('imgs', (Nsteps, *detector.shape), 'f')    
   
    # turn the crank
    t0 = time.time()
    Nsteps = 2
    for i in range(Nsteps):
        mdsim = load_lammps_txt(txtfiles[i], origin=np.array([8, 8, 8]), scale=scale)
        pts = mdsim['POSITIONS']
        img = mdscatter.dft(pts, qvecs)
        img = np.abs(img)**2
        img = np.reshape(img, detector.shape)
        plt.imshow(np.log(img+1), origin='lower')
        plt.show()
        dset[i,:,:] = np.reshape(img, detector.shape)
    t1 = time.time() - t0
    print('time taken = %f\n' % t1)
    h5f.close()
