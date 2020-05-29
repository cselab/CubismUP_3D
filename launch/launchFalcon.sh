#!/bin/bash

SETTINGSNAME=$1
BASENAME=$2
if [ $# -lt 2 ] ; then
  echo "Usage "$0" SETTINGSNAME BASENAME"
  exit 1
fi
BASEPATH=${SCRATCH}/CubismUP3D/

if [ ! -f $SETTINGSNAME ]; then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME

FOLDER=${BASEPATH}${BASENAME}
mkdir -p ${FOLDER}

if [[ ! -z "$icH5Path" ]] && [ ! -z "$icH5File" ]; then
  icFile=${icH5Path}/${icH5File}.h5
  if [ -f ${icFile} ]; then
    cp ${icFile} ${FOLDER}
  else
    echo "${icFile} does not exist"
    exit 0
  fi
fi

cp $SETTINGSNAME ${FOLDER}/settings.sh
[[ -n "${FFACTORY}" ]] && cp ${FFACTORY} ${FOLDER}/factory
cp ../bin/simulation ${FOLDER}

cd $FOLDER

export OMP_NUM_THREADS=12
echo "$OPTIONS" > settings.txt
mpirun -np 1 ./simulation ${OPTIONS} -factory-content "${FACTORY}"
