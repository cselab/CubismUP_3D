#!/bin/bash
SETTINGSNAME=$1
BASENAME=$2
if [ $# -lt 2 ] ; then
  echo "Usage "$0" SETTINGSNAME BASENAME"
  exit 1
fi
BASEPATH=${SCRATCH}/CubismUP3D/

INTERACTIVE=0
NNODE=1
if [ $# -gt 2 ] ; then
    if [ "${3}" = "node" ]; then
        echo "Running on current node"
        INTERACTIVE=1
    else
        NNODE=$3
    fi
fi

NTHREADS=36
if [ $# -gt 3 ] ; then
NTHREADS=$4
fi

if [ ! -f $SETTINGSNAME ];then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME

NPROCESSORS=$((${NNODE}*${NTHREADS}))
FOLDER=${BASEPATH}${BASENAME}
mkdir -p ${FOLDER}

cp $SETTINGSNAME ${FOLDER}/settings.sh
[[ -n "${FFACTORY}" ]] && cp ${FFACTORY} ${FOLDER}/factory
cp ../bin/simulation ${FOLDER}/simulation

cd $FOLDER

unset LSB_AFFINITY_HOSTFILE
export MV2_ENABLE_AFFINITY=0
export OMP_NUM_THREADS=${NTHREADS}
echo $OPTIONS > settings.txt

export LD_LIBRARY_PATH=/cluster/home/novatig/hdf5-1.10.1/gcc_6.3.0_openmpi_2.1/lib/:$LD_LIBRARY_PATH

if [ $INTERACTIVE -eq 1 ] ; then
echo $OPTIONS > settings.txt
#mpirun -n ${NNODE} --map-by ppr:1:node ./simulation ${OPTIONS} -factory-content "${FACTORY}"
#mpirun -n ${NNODE} ./simulation ${OPTIONS} -factory-content "${FACTORY}"
./simulation ${OPTIONS} -factory-content "${FACTORY}"
#mpirun -n ${NNODE} --map-by ppr:1:socket:pe=12 --bind-to core -report-bindings --mca mpi_cuda_support 0 valgrind --tool=memcheck --leak-check=yes --track-origins=yes --show-reachable=yes ./simulation ${OPTIONS}
#mpirun -n ${NNODE} --map-by ppr:1:socket:pe=12 --bind-to core -report-bindings --mca mpi_cuda_support 0 valgrind --tool=memcheck --undef-value-errors=no --num-callers=500  ./simulation ${OPTIONS}
#mpirun -n ${NNODE} --map-by ppr:1:socket:pe=12 --bind-to core -report-bindings --mca mpi_cuda_support 0  ./simulation ${OPTIONS} -factory-content "${FACTORY}"
#mpirun -np ${NNODE} -ppn 1 ./simulation ${OPTIONS}
else
#bsub -R "select[model==XeonE5_2680v3] span[ptile=${NTHREADS}]" -n ${NPROCESSORS} -W 24:00 \
#-J ${BASENAME} "mpirun -n ${NNODE} ./simulation ${OPTIONS} -factory-content "${FACTORY}""
bsub -n ${NPROCESSORS} -W 24:00 -J ${BASENAME} "./simulation ${OPTIONS}"
fi

#valgrind --tool=memcheck --track-origins=yes --leak-check=yes
