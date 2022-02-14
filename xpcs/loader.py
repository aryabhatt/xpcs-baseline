import numpy as np
import os
import glob

"""contributed by Jerome Kieffer from ESRF"""
def load_lammps_bin(fname, origin = np.array([0, 0, 0]), scale =1):
    with open(fname, "rb") as fd:
        raw = fd.read()
    frame_id, natom = np.frombuffer(raw[:16], dtype="uint64")
    size = int(32*natom)
    assert size<len(raw)
    array = np.frombuffer(raw[-size:], dtype=np.float64)
    array.shape = -1, 4
    return (ary[:, 1:] - origin) * scale

def load_npy(npyfile, origin = np.array([0, 0, 0]), scale = 1):
    return (np.load(npyfile).T - origin) * scale


def load_lammps_txt(fname, origin=np.array([0, 0, 0]), scale=1):
    with open(fname,"r") as fp:
       lines = fp.readlines()

    pts = {'TIMESTEP': None, 'NUMBER_OF_ATOMS': None, 'POSITIONS': None}
    for i, line in enumerate(lines):
        if 'TIMESTEP' in line:
            pts['TIMESTEP'] = int(lines[i+1])
        if 'NUMBER OF ATOMS' in line:
            pts['NUMBER_OF_ATOMS'] = int(lines[i+1])
        if 'ATOMS x y z' in line:
            ipos = i+1 

    positions = []
    for i in range(ipos, len(lines)):
        x, y, z = lines[i].split()
        x = (float(x) - origin[0])*scale
        y = (float(y) - origin[1])*scale
        z = (float(z) - origin[2])*scale
        positions.append([x, y, z])

    pts['POSITIONS'] = np.array(positions)
    return pts


def list_lammps_txt_files(path, pattern = None):
    if pattern is not None:
        pat = '/' + pattern
    else:
        pat = '/*.txt'
    txtfiles = glob.glob(path + pat)
    txtfiles.sort(key=os.path.getmtime)
    return txtfiles
 
if __name__ == '__main__':
    path = '../lammps'
    pattern = 'al'
    ffs = lammps_txt(path, pattern)
    print(ffs[10])
