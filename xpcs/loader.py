import numpy as np


"""contributed by Jerome Kieffer from ESRF"""
def load_lammps_bin(fname, center = np.array([0, 0, 0], scale =1):
    with open(fname, "rb") as fd:
        raw = fd.read()
    frame_id, natom = np.frombuffer(raw[:16], dtype="uint64")
    size = int(32*natom)
    assert size<len(raw)
    array = np.frombuffer(raw[-size:], dtype=np.float64)
    array.shape = -1, 4
    return (ary[:, 1:] - center) * scale

def load_npy(npyfile, center = np.array([0, 0, 0]), scale = 1):
    return (np.load(npyfile).T - center) * scale
