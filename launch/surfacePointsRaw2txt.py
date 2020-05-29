# This script works by prividing the path to the various surface.[...].raw
# files up to (and included) the 'rank' token.
# the surface files contain 16 fields (unless modified):
# s, x, y, z, fx, fy, fz, vx, vy, vz, udef(x,y,z), gradchi(x,y,z)
# for all points in the obtacle volume with non-zero grad chi (surface)
import numpy as np
import sys

PATH   =     sys.argv[1]
nRanks = int(sys.argv[2])

TOTAL = np.zeros([0])
for i in range(nRanks):
  FILE = "%s%03d.raw" % (PATH, i)
  DATA = np.fromfile(FILE, dtype=np.float32)
  DATA = DATA.reshape(DATA.size // 16, 16)
  if i==0: TOTAL = DATA
  else: TOTAL = np.append(TOTAL, DATA, axis=0)

np.savetxt(PATH+'.txt', TOTAL, delimiter=',')
