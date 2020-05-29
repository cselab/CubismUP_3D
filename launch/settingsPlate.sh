#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-32}
BPDY=${BPDY:-$((${BPDX}/2))} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/2))} #${BPDZ:-32}

NU=${NU:-0.0002}
BC=${BC:-freespace}

FACTORY='Plate L=0.2 xpos=0.25 xvel=0.1 a=0.2 b=0.4 thickness=0.05 alpha=30 bForcedInSimFrame=1 bFixFrameOfRef=1
'

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.01 -tend 8 "
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -BC_x ${BC} -BC_y ${BC} -BC_z ${BC}"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -use-dlm 10 -nu ${NU}"
