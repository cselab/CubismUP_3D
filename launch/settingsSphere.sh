#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))

DLM=${DLM:-0}
LAMBDA=${LAMBDA:-1e5}

BPDX=${BPDX:-64}
BPDY=${BPDY:-$((${BPDX}/2))} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/2))} #${BPDZ:-32}

#to compare against Mimeau Cottet Mortazavi 2016:
#NU=${NU:-0.00005208333333} # Re 300
NU=${NU:-0.00015625} # Re 100
#BC=${BC:-freespace}
BC=${BC:-dirichlet}

FACTORY='Sphere L=0.125 xpos=0.35 xvel=0.125 bForcedInSimFrame=1 bFixFrameOfRef=1
'
# for accel and decel start and stop add accel=1 T=time_for_accel
# shift center to shed vortices immediately by ypos=0.250244140625 zpos=0.250244140625

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.5 -tend 12 -iterativePenalization 1 "
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -BC_x ${BC} -BC_y ${BC} -BC_z ${BC}"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -lambda ${LAMBDA} -use-dlm ${DLM} -nu ${NU}"
