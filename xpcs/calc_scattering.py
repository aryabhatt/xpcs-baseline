#! /usr/bin/env python

import mdscatter
import numpy as np
import os
import re
import h5py
import time

N = 512

def filelist(path, pattern = None):
    if m is None:
        m = re.compile(pattern)
    data_files = [ os.path.join(path, f) for f in os.listdir(path) if m.search(f) ]
    return sorted(data_files)

def qvals(theta = [-10, 10], wavelen = 0.1, nrow = 64, ncol = 64):
    th = np.deg2rad(np.linspace(theta[0], theta[1], ncol))
    al = np.deg2rad(np.linspace(theta[0], theta[1], ncol))
    th, al = np.meshgrid(th, th)

    if wavelen is None:
        wavelen = 0.1
    k0 = 2 * np.pi / wavelen
    qx = k0 * (np.cos(al) * np.cos(th) - 1)
    qy = k0 * (np.cos(al) * np.sin(th))
    qz = k0 * np.sin(al)
    qpts = np.array([qx.ravel(), qy.ravel(), qz.ravel()]).T
    return qpts

if __name__ == '__main__':
    wavelen = 0.1
    qvals = qvals(nrow = N, ncol = N)

    outf = 'xpcs' + str(N).zfill(5) + '.h5'
    h5f = h5py.File(outf, 'w')
    grp = h5f.create_group('xpcs')
    qtmp = grp.create_dataset('q_points', (3, N*N), 'f')
    qtmp.attrs['wavelen'] = wavelen
    qtmp.attrs['theta'] = [-10, 10]
    qtmp.attrs['theta_units'] = 'degree'

    # read data
    pattern = '[3-5](\d){4}'
    npys = filelist('data', pattern)
    Nsteps = len(npys)
    dset = grp.create_dataset('imgs', (Nsteps, N, N), 'f')    

    # turn the crank
    t0 = time.time()
    for i, npy in enumerate(npys):
        pts = np.load(npy).T 
        img = mdscatter.dft(pts, qvals)
        img = np.abs(img)**2
        dset[i,:,:] = np.reshape(img, (N, N))
    t1 = time.time() - t0
    print('time taken = %f\n' % t1)
    h5f.close()
