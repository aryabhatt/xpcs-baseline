#! /usr/bin/env python

import mdscatter
import numpy as np
import os
import re
import h5py
import time
from loader import load_npy
from detector import Lambda750k, Square512

def filelist(path, pattern = None):
    if not os.path.isdir(path):
        raise ValueError('illegal data path')
    m = re.compile(pattern)
    data_files = [ os.path.join(path, f) for f in os.listdir(path) if m.search(f) ]
    return sorted(data_files)

if __name__ == '__main__':

    wavelen = 0.1127
    energy = 1.23984 / wavelen
    
    sdd = 4.
    scale = 28
    center = (0, 768)

    beam_rad = 6 * scale
    detector = Lambda750k()
    qvecs = detector.qvectors(sdd, center, wavelen)

    outf = 'xpcs_out.h5'
    h5f = h5py.File(outf, 'w')
    grp = h5f.create_group('xpcs')
    qtmp = grp.create_dataset('q_points', (3, *detector.shape), 'f')

    # read data
    pattern = '3(\d){4}'
    npys = filelist('/home/dkumar/Data/np_arrays', pattern)
    Nsteps = len(npys)
    dset = grp.create_dataset('imgs', (Nsteps, *detector.shape), 'f')    

    # turn the crank
    t0 = time.time()
    for i, npy in enumerate(npys):
        pts = load_npy(npy, center = np.array([8, 8, 8]), scale = scale)
        img = mdscatter.dft(pts, qvals, beam_rad)
        img = np.abs(img)**2
        img = np.reshape(img, detector.shape)
        dset[i,:,:] = np.reshape(img, detector.shape)
    t1 = time.time() - t0
    print('time taken = %f\n' % t1)
    h5f.close()
