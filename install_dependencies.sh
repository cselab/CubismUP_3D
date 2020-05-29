#!/bin/bash

# TODO: sha256 and md5 check for Mac.
# TODO: For each dependency, put the installation steps in a function.
# TODO: Update $SOURCES to $PWD/source/dependencies or something.


set -e

# https://stackoverflow.com/a/23378780/2203044
# LOGICAL_CORE_COUNT=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.logicalcpu_max || lscpu -p | egrep -v '^#' | wc -l)
PHYSICAL_CORE_COUNT=$([[ $(uname) = 'Darwin' ]] && sysctl -n hw.physicalcpu_max || lscpu -p | egrep -v '^#' | sort -u -t, -k 2,4 | wc -l)

# Parameters modifiable from environment.
JOBS=${JOBS:-$PHYSICAL_CORE_COUNT}
SOURCES=${SOURCES:-$PWD/dependencies}
INSTALL_PATH=${INSTALL_PATH:-$PWD/dependencies/build}

# Shorthands for versions.
# NOTE: Avoid updating the version (of CMake) because on clusters it may
#       require manual installation, which means adding ~15k files!
# NOTE: Changing these numbers may not be enough for the script to work properly!
# NOTE: Update `CMakeLists.txt` and `README.md` if updating versions!
ACCFFT_GIT_URL='https://github.com/novatig/accfft'
ACCFFT_COMMIT='b88432cd71118b7997fa67948cb32125cb881fc1'

CMAKE_MINIMUM_REQUIRED_VERSION=3.2  # See `CMakeLists.txt`.
CMAKE_VERSION=3.11.1
CMAKE_VERSION_SHORT=3.11
CMAKE_SHA_256='57bebc6ca4d1d42c6385249d148d9216087e0fda57a47dc5c858790a70217d0c  cmake-3.11.1.tar.gz'

FFTW_VERSION=3.3.7
FFTW_MD5='0d5915d7d39b3253c1cc05030d79ac47  fftw-3.3.7.tar.gz'

HDF5_VERSION=1.10.1
HDF5_MD5='43a2f9466702fb1db31df98ae6677f15  hdf5-1.10.1.tar.gz'
HDF5_URL='https://www.hdfgroup.org/package/source-gzip/?wpdmdl=4301&refresh=5afee8d8a45151526655192'

GSL_VERSION=2.1

# Other shorthands.
TAR="tar --keep-newer-files"

# Flags. By default all are disabled.
INSTALL_ACCFFT=
INSTALL_CMAKE=
INSTALL_FFTW=
INSTALL_HDF5=
INSTALL_GSL=
UNKNOWN_ARGUMENT=
PRINT_EXPORT=

# Determine what the user wants.
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -a|--all) INSTALL_ACCFFT=1; INSTALL_CMAKE=1; INSTALL_FFTW=1; INSTALL_HDF5=1; INSTALL_GSL=1; shift ;;
        -e|--export) PRINT_EXPORT=1; shift ;;
        --accfft) INSTALL_ACCFFT=1; shift ;;
        --cmake) INSTALL_CMAKE=1; shift ;;
        --fftw) INSTALL_FFTW=1; shift ;;
        --hdf5) INSTALL_HDF5=1; shift ;;
        --gsl) INSTALL_GSL=1; shift ;;
        *) UNKNOWN_ARGUMENT=1; shift ;;
    esac
done


if [ -z "$INSTALL_ACCFFT" -a -z "$INSTALL_CMAKE" -a -z "$INSTALL_FFTW" -a -z "$INSTALL_HDF5" -a -z "$INSTALL_GSL" -a -z "$PRINT_EXPORT" -o -n "$UNKNOWN_ARGUMENT" ]; then
    echo "Usage:
    ./install_dependencies [-a | --all | [[--accfft] [--cmake] [--fftw] [--hdf5] [--gsl]]]

Arguments:
  -a,  --all    - Install all available libraries and tools
  -e,  --export - Print export commands for all libraries and tools
                  (assuming they are installed)
  --accfft      - Install AccFFT
  --cmake       - Install CMake ${CMAKE_VERSION} (required at least ${CMAKE_MINIMUM_REQUIRED_VERSION})
  --fftw        - Install FFTW ${FFTW_VERSION}
  --hdf5        - Install HDF5 ${HDF5_VERSION}
  --gsl         - Install GSL ${GSL_VERSION}

All libraries and tools are installed locally in the dependencies/ folder.
Note that this script tries not to redo everything from scratch if run multiple
times. Therefore, in case of errors, try erasing the dependencies/ folder.

Note: AccFFT requires FFTW as a dependency!
"
    exit
fi


BASEPWD=$PWD


if [ -n "$INSTALL_CMAKE" ]; then
    echo "Installing CMake ${CMAKE_VERSION}..."
    wget -nc https://cmake.org/files/v${CMAKE_VERSION_SHORT}/cmake-${CMAKE_VERSION}.tar.gz -P $SOURCES
    cd $SOURCES
    [ -x "$(command -v sha256sum)" ] && sha256sum --quiet -c - <<< $CMAKE_SHA_256
    $TAR -xzvf cmake-${CMAKE_VERSION}.tar.gz
    cd cmake-${CMAKE_VERSION}
    ./bootstrap --parallel=${JOBS} --prefix=$INSTALL_PATH/cmake-${CMAKE_VERSION}/
    make -j${JOBS}
    make install -j${JOBS}
    cd $BASEPWD
fi


if [ -n "$INSTALL_FFTW" ]; then
    echo "Installing FFTW ${FFTW_VERSION}..."
    wget -nc http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz -P $SOURCES
    cd $SOURCES
    [ -x "$(command -v md5sum)" ] && md5sum --quiet -c - <<< $FFTW_MD5
    $TAR -xzvf fftw-${FFTW_VERSION}.tar.gz
    cd fftw-${FFTW_VERSION}
    FFTW_ROOT=$INSTALL_PATH/fftw-${FFTW_VERSION}/
    ./configure --prefix=$FFTW_ROOT --enable-mpi --enable-openmp --enable-shared
    make -j${JOBS}
    make install -j${JOBS}
    cd $BASEPWD
fi


if [ -n "$INSTALL_ACCFFT" ]; then
    echo "Installing AccFFT ${CMAKE_VERSION}..."
    if [ -n "${FFTW_ROOT}" ]; then
        : # OK
    elif [ -n "${FFTW_DIR}" ]; then
        FFTW_ROOT=${FFTW_DIR}/../
    elif [ -n "${FFTW3_ROOT_DIR}" ]; then
        FFTW_ROOT=${FFTW3_ROOT_DIR}
    else
        echo "FFTW not found, aborting. None of the following environment variables set:"
        echo "    FFTW_ROOT, FFTW_DIR, FFTW3_ROOT_DIR"
        exit 1
    fi
    mkdir -p $SOURCES
    cd $SOURCES
    rm -rf accfft
    git clone ${ACCFFT_GIT_URL} accfft
    cd accfft
    git checkout ${ACCFFT_COMMIT}
    mkdir build
    cd build
    # For FFTW_ROOT below, note that CMake variables are not the same as env variables.
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH/accfft/ \
          -DFFTW_ROOT="$FFTW_ROOT" \
          -DBUILD_GPU=true \
          -DBUILD_SHARED=false \
          ..
    make -j $JOBS
    make install
    cd $BASEPWD
fi


if [ -n "$INSTALL_HDF5" ]; then
    echo "Installing HDF5 ${HDF5_VERSION}..."
    mkdir -p $SOURCES
    wget ${HDF5_URL} -O $SOURCES/hdf5-${HDF5_VERSION}.tar.gz
    cd $SOURCES
    [ -x "$(command -v md5sum)" ] && md5sum --quiet -c <<< $HDF5_MD5
    $TAR -xzvf hdf5-${HDF5_VERSION}.tar.gz
    cd hdf5-${HDF5_VERSION}
    CC=mpicc ./configure --prefix=$INSTALL_PATH/hdf5-${HDF5_VERSION}-parallel/ --enable-parallel
    make -j $JOBS
    make install -j $JOBS
    cd $BASEPWD
fi


if [ -n "$INSTALL_GSL" ]; then
    echo "Installing GSL ${GSL_VERSION}..."
    wget -nc ftp://ftp.gnu.org/gnu/gsl/gsl-${GSL_VERSION}.tar.gz -P $SOURCES
    cd $SOURCES
    $TAR -xzvf gsl-${GSL_VERSION}.tar.gz
    cd gsl-${GSL_VERSION}
    ./configure --prefix=$INSTALL_PATH/gsl-${GSL_VERSION} --enable-parallel
    make -j $JOBS
    make install -j $JOBS
    cd $BASEPWD
fi

if [ -n "$INSTALL_CMAKE" ]; then
    echo
    echo "======================================================================"
    echo "Done! Run or add to ~/.bashrc the following command:"
    echo
fi
if [ -n "$INSTALL_CMAKE" -o -n "$PRINT_EXPORT" ]; then
    echo "export PATH=$INSTALL_PATH/cmake-${CMAKE_VERSION}/bin:\$PATH"
fi
