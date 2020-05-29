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

NPROCESSORS=$((${NNODE}*12))
FOLDER=${BASEPATH}${BASENAME}
mkdir -p ${FOLDER}

cp $SETTINGSNAME ${FOLDER}/settings.sh
[[ -n "${FFACTORY}" ]] && cp ${FFACTORY} ${FOLDER}/factory
cp ../bin/${EXEC} ${FOLDER}/simulation
cp -r ../source ${FOLDER}/
cp $0 ${FOLDER}

git diff HEAD > ${FOLDER}/gitdiff.diff

cd ${FOLDER}

cat <<EOF >daint_sbatch
#!/bin/bash -l

#SBATCH --account=s929
#SBATCH --job-name="${BASENAME}"
#SBATCH --output=${BASENAME}_out_%j.txt
#SBATCH --error=${BASENAME}_err_%j.txt

#SBATCH --time=${WCLOCK}
#SBATCH --partition=${PARTITION}
#SBATCH --constraint=gpu

#SBATCH --nodes=${NNODE}
#SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=12
# #SBATCH --threads-per-core=1
# #SBATCH --mail-user=${MYNAME}@ethz.ch
# #SBATCH --mail-type=ALL

export MPICH_MAX_THREAD_SAFETY=multiple
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_NUM_THREADS=12

srun --ntasks ${NNODE} --ntasks-per-node=1 ./simulation ${OPTIONS} -factory-content $(printf "%q" "${FACTORY}")

EOF

chmod 755 daint_sbatch
sbatch daint_sbatch

#srun --ntasks ${NNODE} --threads-per-core=1 --ntasks-per-node=1 --cpus-per-task=12 time ./simulation ${OPTIONS} -factory-content $(printf "%q" "${FACTORY}")
