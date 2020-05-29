#!/bin/bash
NNODEX=${NNODEX:-1}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))

#BPDX=${BPDX:-16}
BPDX=${BPDX:-12}
BPDY=${BPDY:-${BPDX}} #${BPDY:-32}
BPDZ=${BPDZ:-${BPDX}} #${BPDZ:-32}

NU=${NU:-0.005}
#TKE=${TKE:-0}
#TKE0=${TKE0:-0.6}
EPS=${EPS:-0.1}
EXT=${EXT:-6.2831853072}
TANALYSIS=${TANALYSIS:-0.1}

FACTORY=''

OPTIONS=
OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -extentx ${EXT}"
OPTIONS+=" -dump2D 0 -dump3D 0 -tdump 1"
OPTIONS+=" -BC_x periodic -BC_y periodic -BC_z periodic"
OPTIONS+=" -initCond HITurbulence"
OPTIONS+=" -spectralForcing 1"

OPTIONS+=" -spectralIC fromFit"
#OPTIONS+=" -spectralIC art -tke0 ${TKE0} -k0 4"
OPTIONS+=" -energyInjectionRate ${EPS}"
#OPTIONS+=" -turbKinEn_target ${TKE}"

OPTIONS+=" -nprocsx ${NNODEX} -nprocsy 1 -nprocsz 1"
OPTIONS+=" -CFL 0.02 -tend 100 -compute-dissipation 1"
OPTIONS+=" -analysis HIT -tAnalysis ${TANALYSIS}"
OPTIONS+=" -keepMomentumConstant 1"
OPTIONS+=" -nu ${NU}"
