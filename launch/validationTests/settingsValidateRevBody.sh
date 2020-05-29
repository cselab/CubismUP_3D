#!/bin/bash

# Cd sim should be ~ 0.32
# Shape of body = rotationally symmetric NACA0030. (just use the Deadset height=width=NACA0030 in FishLibrary)

# Cd theoretical = 0.332. Go check Hoerner's "Fluid Dynamic Drag", Pg 6-16, Eq 23. 
# Use d/l=0.3 and C_flam = 0.02 from Fig 22 close to Re=3000.

BASENAME=bodyOfRevolution_fishSpeed
NNODE=32
NNODEX=32
NNODEY=1
WCLOCK=24:00:00
WSECS=43000
FFACTORY=factoryBodyRev

OPTIONS=
OPTIONS+=" -uinfx 1.44"
OPTIONS+=" -bpdx 128 -bpdy 64 -bpdz 32"
OPTIONS+=" -2Ddump 0 -restart 0"
OPTIONS+=" -nprocsx ${NNODEX}"
OPTIONS+=" -nprocsy ${NNODEY}"
OPTIONS+=" -nprocsz 1"
OPTIONS+=" -CFL 0.1"
OPTIONS+=" -length 0.2"
OPTIONS+=" -lambda 1e6"
OPTIONS+=" -nu 0.00008"
OPTIONS+=" -tend 3.0 -tdump 0.2"
