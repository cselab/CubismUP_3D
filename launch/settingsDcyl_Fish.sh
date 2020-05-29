#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-32}
BPDY=${BPDY:-$((${BPDX}/2))} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/4))} #${BPDZ:-32}

NU=${NU:-0.00001}

FACTORY='Cylinder L=0.1 xpos=0.2 ypos=0.25 zpos=0.0625 xvel=0.3 bFixFrameOfRef=1 bForcedInSimFrame=1 section=D
StefanFish L=0.3 T=1 xpos=0.45 bFixToPlanar=1 bForcedInSimFrame=1
'

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.1 -tend 20 "
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -BC_x dirichlet -BC_y dirichlet -BC_z periodic"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -use-dlm 10 -nu ${NU}"
