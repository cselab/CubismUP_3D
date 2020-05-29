#!/bin/bash
SETTINGSNAME=$1
BASENAME=$2
if [ $# -lt 2 ] ; then
  echo "Usage "$0" SETTINGSNAME BASENAME"
  exit 1
fi

WCLOCK=${WCLOCK:-24:00:00}
PARTITION=${PARTITION:-normal}
EXEC=${EXEC:-simulation}

MYNAME=`whoami`
BASEPATH="${SCRATCH}/CubismUP3D/"
#lfs setstripe -c 1 ${BASEPATH}${RUNFOLDER}

if [ ! -f $SETTINGSNAME ]; then
    echo ${SETTINGSNAME}" not found! - exiting"
    exit -1
fi
source $SETTINGSNAME

FOLDER=${BASEPATH}${BASENAME}
mkdir -p ${FOLDER}

cp $SETTINGSNAME ${FOLDER}/settings.sh
[[ -n "${FFACTORY}" ]] && cp ${FFACTORY} ${FOLDER}/factory
cp ../bin/${EXEC} ${FOLDER}/simulation
cp -r ../source ${FOLDER}/
cp $0 ${FOLDER}

git diff HEAD > ${FOLDER}/gitdiff.diff

cd ${FOLDER}

NNODE=$(($NPROCX/2))

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=eth2
#SBATCH --job-name="${BASENAME}"
#SBATCH --output=${BASENAME}_out_%j.txt
#SBATCH --error=${BASENAME}_err_%j.txt

#SBATCH --time=${WCLOCK}
#SBATCH --partition=${PARTITION}

#SBATCH --nodes=${NNODE}
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=18
#SBATCH --threads-per-core=1
#SBATCH --constraint=mc
#SBATCH --mail-user=${MYNAME}@ethz.ch
#SBATCH --mail-type=ALL

module swap daint-gpu daint-mc
module swap gcc gcc/7.1.0
module load GSL cray-hdf5-parallel
module load fftw

export MPICH_MAX_THREAD_SAFETY=multiple
export OMP_NUM_THREADS=18
srun --ntasks ${NPROCX} --threads-per-core=1 --ntasks-per-node=2 --cpus-per-task=18 time ./simulation ${OPTIONS} -factory-content $(printf "%q" "${FACTORY}")
EOF

chmod 755 daint_sbatch
sbatch daint_sbatch
