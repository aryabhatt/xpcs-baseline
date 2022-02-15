"""
Matplotlib Animation Example

author: Jake Vanderplas
email: vanderplas@astro.washington.edu
website: http://jakevdp.github.com
license: BSD
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import h5py

# open HDF5 File
if len(sys.argv) == 2:
    h5f = sys.argv[1]
    if h5f.endswith('h5') and os.path.isfile(h5f):
        fp = h5py.File(h5f, 'r')
        qvec = fp['gixpcs/q_points']
        dset = fp['gixpcs/imgs']
    else:
        print('file %s not found' % h5f)
        exit(1)
else:
    print('Usage: python %s <h5 file>' % sys.argv[0])
    exit(1)

qp = np.sign(qvec[:,:,1]) * np.sqrt(qvec[:,:,0]**2 + qvec[:,:,1]**2)
xn, xp = (qp.min(), qp.max())
yn, yp = (qvec[:,:,2].min(), qvec[:,:,2].max())
lims = [xn, xp, yn, yp]

nz, ny, nx = dset.shape


fig, ax = plt.subplots()

ims = []
for i in range(1, nz):
    im = ax.imshow(np.log(dset[i,:,:]+1), animated=True, origin='lower', extent=lims)
    if i == 1:
        ax.imshow(np.log(dset[i,:,:]+1), origin='lower', extent=lims)
    ims.append([im])

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.ArtistAnimation(fig, ims, interval=20, blit=True, repeat_delay=1000)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
moviename = os.path.splitext(h5f)[0] + '.mp4'
anim.save(moviename, fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
