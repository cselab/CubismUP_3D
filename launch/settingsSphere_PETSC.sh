#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-32}
BPDY=${BPDY:-$((${BPDX}/2))} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/2))} #${BPDZ:-32}

NU=${NU:-0.00015625} # Re 100

FACTORY='Sphere L=0.125 xpos=0.3 xvel=0.125 bForcedInSimFrame=1 bFixFrameOfRef=1
'
# for accel and decel start and stop add accel=1 T=time_for_accel
# shift center to shed vortices immediately by ypos=0.250244140625 zpos=0.250244140625

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.5 -tend 8 "
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -BC_x dirichlet -BC_y dirichlet -BC_z dirichlet"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -nu ${NU} -useSolver petsc"
