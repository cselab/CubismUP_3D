#!/bin/bash
NNODEX=${NNODEX:-4}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))

BPDX=${BPDX:-32}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-$((${BPDX}/8))} #${BPDZ:-32}

NU=${NU:-0.1} # RE = 8/12 Ly Umax / \nu = 66.666

FACTORY=''

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -dump2D 0 -dump3D 1 -tdump 0.1"
OPTIONS+=" -BC_x periodic -BC_y wall -BC_z periodic -initCond channel"
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -CFL 0.1 -tend 10 -uMax_forced 1 -compute-dissipation 1"
OPTIONS+=" -nu ${NU}"
