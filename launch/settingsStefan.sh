#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))
DLM=${DLM:-1}
CFL=${CFL:-0.1}
LAMBDA=${LAMBDA:-1e4}
BPDX=${BPDX:-32}
BPDY=${BPDY:-$((${BPDX}/2))} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/2))} #${BPDZ:-32}

#to compare against Wim's thesis:
NU=${NU:-0.0004545454545} # Re 550
#NU=${NU:-0.0000224} # Re 550
BC=${BC:-freespace}
#BC=${BC:-dirichlet}

#FACTORY='CarlingFish L=0.3 T=1 xpos=0.3 bFixToPlanar=1 bFixFrameOfRef=1 Correct=1
#FACTORY='StefanFish L=0.3 T=1 xpos=0.3 bFixToPlanar=1 bFixFrameOfRef=1 Correct=1
FACTORY='CarlingFish L=0.5 T=1.0 xpos=0.4 bFixToPlanar=1 bFixFrameOfRef=1 heightProfile=stefan widthProfile=stefan
'

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ} -restart 1"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.5 -tend 6 -freqDiagnostics 10 "
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -BC_x ${BC} -BC_y ${BC} -BC_z ${BC} -iterativePenalization 1"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL ${CFL} -lambda ${LAMBDA} -use-dlm ${DLM} -nu ${NU}"
