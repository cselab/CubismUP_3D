#!/bin/bash
NNODEX=${NNODEX:-1}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-4}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/2))} #${BPDZ:-32}

NU=${NU:-0.00035714} # RE = halfHeight * U / nu = 2800

FACTORY=''

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 1 -dump3D 0 -tdump 10"
OPTIONS+=" -BC_x periodic -BC_y wall -BC_z periodic -fadeLen 0 -initCond channelRandom"
OPTIONS+=" -nslices 2 -slice1_direction 1 -slice2_direction 2 "
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"

OPTIONS+=" -extentx 6.2831853072 -extenty 2 -extentz 4.7123889804"

OPTIONS+=" -mesh_density_y SinusoidalDensity -eta_y 0.95" # first mesh point at 0.012523878 (Kim and Moin 1987)
#OPTIONS+=" -iterativePenalization 0 -useSolver petsc"
OPTIONS+=" -iterativePenalization 0 -useSolver hypre"

# BAD : stuff to test with iterative penalization (almost uniform grid):
#OPTIONS+=" -extentx 6.2831853072 -extenty 2 -extentz 4.7123889804"
#OPTIONS+=" -mesh_density_y SinusoidalDensity -eta_y 0.9" # first mesh point at y+ = 0.018313569
#OPTIONS+=" -iterativePenalization 1"

OPTIONS+=" -fixedMassFlux 1 -sgs 0"
OPTIONS+=" -CFL 0.1 -tend 100 -uMax_forced 1.5 -compute-dissipation 1 -nu ${NU}"
