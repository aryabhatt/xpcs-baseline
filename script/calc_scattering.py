#! /usr/bin/env python

import mdscatter
import numpy as np
import os
import re
import h5py
import time
import glob

N = 512

def read_lammps_bin(fname):
    with open(fname, "rb") as fd:
        raw = fd.read()
    frame_id, natom = np.frombuffer(raw[:16], dtype="uint64")
    size = int(32*natom)
    assert size<len(raw)
    ary = np.frombuffer(raw[-size:], dtype=np.float64)
    ary.shape = -1, 4
    return ary[:, 1:]


def filelist(path, pattern = None):
    if pattern is None:
        data_files = [ os.path.join(path, f) for f in os.listdir(path)]
    else:
        m = re.compile(pattern)
        data_files = [ os.path.join(path, f) for f in os.listdir(path) if m.search(f) ]
    return sorted(data_files)


def qvals(theta = [-10, 10], wavelen = 0.1, nrow = 64, ncol = 64):
    
    th = np.deg2rad(np.linspace(theta[0], theta[1], nrow))
    al = np.deg2rad(np.linspace(theta[0], theta[1], ncol))
    th, al = np.meshgrid(th, al)

    k0 = 2 * np.pi / wavelen
    qx = k0 * (np.cos(al) * np.cos(th) - 1)
    qy = k0 * (np.cos(al) * np.sin(th))
    qz = k0 * np.sin(al)
    qpts = np.array([qx.ravel(), qy.ravel(), qz.ravel()]).T
    return qpts

if __name__ == '__main__':
    wavelen = 0.1
    q_points = qvals(wavelen = wavelen, nrow = N, ncol = N)
    pattern = '[3-5](\d){4}'
    binfiles = glob.glob("data/*.bin")
    if not binfiles:
        raise RuntimeError("Please put the lammps simulation in the data folder")
    Nsteps = len(binfiles)
    binfiles.sort(key=lambda name: int(name.split(".")[1]))
    outf = 'xpcs' + str(N).zfill(5) + '.h5'
    with h5py.File(outf, 'w') as h5f:
        grp = h5f.create_group('xpcs')
        qtmp = grp.create_dataset('q_points', (3, N, N), 'f',
                                  chunks=(1, N, N),
                                  compression="gzip", 
                                  shuffle=True)
        qtmp.attrs['wavelen'] = wavelen
        qtmp.attrs['theta'] = [-10, 10]
        qtmp.attrs['theta_units'] = 'degree'
        qtmp.attrs['content'] = 'qx, qy, qz'
        qtmp[0, :, :] = q_points[..., 0].reshape((N, N))
        qtmp[1, :, :] = q_points[..., 1].reshape((N, N))
        qtmp[2, :, :] = q_points[..., 2].reshape((N, N))
        # read data
        dset = grp.create_dataset('imgs', (Nsteps, N, N), 'f',
                                  chunks= (1, N, N),
                                  compression="gzip",
                                  shuffle=True)    
    
        # turn the crank
        t0 = time.time()
        for i, fname in enumerate(binfiles):
            print(fname)
            pts = read_lammps_bin(fname)
            img = mdscatter.dft(pts, q_points)
            img = np.abs(img)**2
            dset[i,:,:] = np.reshape(img, (N, N))
        t1 = time.time() - t0
        print('time taken = %f\n' % t1)
