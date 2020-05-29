#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-64}
BPDY=${BPDY:-$((${BPDX}/2))} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/8))} #${BPDZ:-32}

NU=${NU:-0.000008}
BC=${BC:-freespace}

# Naca in this case is forced. No heaving / pitching.
# airfoil moves with speed xvel and sim box moves with airfoil if bFixFrameOfRef_x
FACTORY='Naca L=0.2 thickness=0.15 xpos=0.25 xvel=0.2 bFixFrameOfRef_x=1 bForcedInSimFrame=1
'

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.05 -tend 20 "
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -BC_x dirichlet -BC_y dirichlet -BC_z periodic"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -use-dlm 10 -nu ${NU}"
