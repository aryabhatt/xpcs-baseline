#! /usr/bin/env python

import mdscatter
import numpy as np
import h5py
import time
import os

from loader import list_lammps_txt_files, load_lammps_txt
from detector import Lambda750k

if __name__ == '__main__':

    wavelen = 0.1127
    energy = 1.23984 / wavelen
    
    sdd = 4.
    scale = 28
    center = (0, 768)

    # read data
    datadir = '../lammps'
    pattern = 'al.*.txt'
    txtfiles = list_lammps_txt_files(datadir, pattern)
    Nsteps = len(txtfiles)
 
    # load detector
    detector = Lambda750k()
    qvecs = detector.qvectors(sdd, center, wavelen)

    # output hdf5 file
    outf = 'xpcs.h5'
    h5f = h5py.File(outf, 'w')
    grp = h5f.create_group('xpcs')
    dset = grp.create_dataset('imgs', (Nsteps, *detector.shape), 'f')    
    qtmp = grp.create_dataset('q_points', (*detector.shape, 3), 'f')
    qtmp[:] = qvecs.reshape(*detector.shape, 3)

      
    # turn the crank
    t0 = time.time()
    for i in range(Nsteps):
        mdsim = load_lammps_txt(txtfiles[i], origin=np.array([8, 8, 8]), scale=scale)
        pts = mdsim['POSITIONS']
        img = mdscatter.dft(pts, qvecs)
        img = np.abs(img)**2
        img = np.reshape(img, detector.shape)
        dset[i,:,:] = np.reshape(img, detector.shape)
    t1 = time.time() - t0
    print('time taken = %f\n' % t1)
    h5f.close()
