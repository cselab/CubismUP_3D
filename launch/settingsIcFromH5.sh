#!/bin/bash
NNODEX=${NNODEX:-1}
NNODEY=1
NNODE=$(($NNODEX * $NNODEY))


icH5Path=${icH5Path:-~/icGenerator/Output/hit_re90/}
icH5File=${icH5File:-downsample_4_4_4}

EXTRASETTINGS=${icH5Path}
source ${icH5Path}/${icH5File}_settings.sh

FACTORY=''

OPTIONS=
OPTIONS+=" -nprocsx ${NNODEX} -nprocsy ${NNODEY} -nprocsz 1"
OPTIONS+=" -dump2D 0 -dump3D 0 -tdump 0"

OPTIONS+=" -bpdx ${BPDX} -bpdy ${BPDY} -bpdz ${BPDZ}"
OPTIONS+=" -BC_x ${BC_x} -BC_y ${BC_y} -BC_z ${BC_z}"
OPTIONS+=" -extentx ${EXTENT_x}"

OPTIONS+=" -icFromH5 $icH5File"
OPTIONS+=" -spectralForcing 1 -sgs SSM -cs 0.2"
OPTIONS+=" -analysis HIT -tAnalysis 0.1"
OPTIONS+=" -CFL 0.1 -tend 20 -nu ${NU}"

mkdir -p ${BASEPATH}${BASENAME}

if [[ ! -z "$icH5Path" ]] && [ ! -z "$icH5File" ]; then
  icFile=${icH5Path}/${icH5File}.h5
  if [ -f ${icFile} ]; then
    cp ${icFile} ${FOLDER}
  else
    echo "${icFile} does not exist"
    exit 0
  fi
fi


