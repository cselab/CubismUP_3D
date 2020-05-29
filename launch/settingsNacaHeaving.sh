#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-64}
BPDY=${BPDY:-$((${BPDX}/2))} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/8))} #${BPDZ:-32}

NU=${NU:-0.00002755428299}

# this is a heaving and pitching naca, free to move in the x direction from the fluid forces
# sim box moves with the x center of mass of the airfoil
# Fheave and Fpitch are frequencies of the two motions
# Apitch and Aheave (non dimensional) are the amplitudes of the two motions
# Ppitch is the phase, divided by 2*M_PI (therefore if Ppitch=0.5 they are antiphase)
# Mpitch is the mean pitch. if Mpitch=0.1 the pitching motion is sinusoidal around 0.1 radians
FACTORY='Naca L=0.2 thickness=0.12 xpos=0.25 xvel=0.15155 Fheave=0.16165 Aheave=0.75 Fpitch=0.16165 Apitch=0.5235987 Ppitch=0.208333 Mpitch=0 bForcedInSimFrame=1 bFixFrameOfRef_x=1 bFixFrameOfRef_y=1
'

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.008 -tend 8 "
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -BC_x dirichlet -BC_y dirichlet -BC_z periodic"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -use-dlm 10 -nu ${NU}"
