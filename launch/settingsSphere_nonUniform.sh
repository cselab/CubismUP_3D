#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=${NNODEY:-1}
NNODE=$(($NNODEX * $NNODEY))

LAMBDA=${LAMBDA:-1e4}

BPDX=${BPDX:-16}
BPDY=${BPDY:-${BPDX}}
BPDZ=${BPDZ:-${BPDX}}

NU=${NU:-0.0004} # Re 300

FACTORY='Sphere L=0.5 xpos=1 xvel=1 bForcedInSimFrame=1 bFixFrameOfRef=1
'
# for accel and decel start and stop add accel=1 T=time_for_accel
# shift center to shed vortices immediately by ypos=0.250244140625 zpos=0.250244140625

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -extentx 4 -extenty 2 -extentz 4"
OPTIONS+=" -mesh_density_y SinusoidalDensity -eta_y 0.95"
OPTIONS+=" -BC_x dirichlet -BC_y wall -BC_z dirichlet -fadeLen 0 -initCond channelRandom"
OPTIONS+=" -dump2D 1 -dump3D 1 -tdump 0.5 -tend 8 "
#OPTIONS+=" -iterativePenalization 0 -useSolver hypre"
OPTIONS+=" -iterativePenalization 0 -useSolver petsc"
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -lambda ${LAMBDA} -use-dlm 0 -nu ${NU}"
