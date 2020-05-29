<!--
TO VIEW THIS FILE, RUN THE FOLLOWING:
    python3 -m pip install --user grip
    python3 -m grip README.md --export README.html

AND OPEN
    README.html

OR USE WEB SERVER VARIANT (NOTE: 60 UPDATES/HOUR LIMIT!!)
    python3 -m grip README.md
-->

# Table of contents

* [Installation](#installation)
* [Detailed installation instructions](#detailed-installation-instructions)
  * [Cluster-specific modules](#cluster-specific-modules)
  * [Installing dependencies manually](#installing-dependencies-manually)
    * [Other options and further information](#other-options-and-further-information)
  * [Advanced solver options (compile time)](#advanced-solver-options-compile-time)
  * [Compiling on Mac](#compiling-on-mac)
  * [Troubleshooting](#troubleshooting)
* [Configuring and running simulations](#configuring-and-running-simulations)
  * [Quick introduction](#quick-introduction)
  * [Detailed description](#detailed-description)
* [Visualization using Paraview](#visualization-using-paraview)
  * [Visualizing shapes](#visualizing-shapes)
  * [Visualizing flow](#visualizing-flow)

# Installation

This repository contains submodules, clone with:
```bash
git clone --recursive https://github.com/cselab/CubismUP_3D.git
```

To install Python dependencies, run:
```bash
python3 -m pip install --user -r requirements.txt
```

To compile, run:
```bash
mkdir -p build
cd build
cmake ..
make
```

This process should take few minutes. If that doesn't work, read the [Detailed installation instructions](#detailed-installation-instructions) below.

# Tests

To run the tests, starting from the repository root folder, execute the following:
```
cd build
ctest
pytest ../tests/python/
```

# Detailed installation instructions

CubismUP uses CMake to automatically detect required dependencies and to compile the code.
If the dependencies are missing, they can be easily downloaded, compiled and locally installed using the provided script `install_dependencies.sh` (details below).
If the dependencies are already available, but CMake does not detect them, appropriate environment variables specifying their path have to be defined.

CubismUP requires the following 3rd party libraries:

| Dependency            | Environment variable pointing to the existing installation |
|-----------------------|-----------------------------------------------|
| FFTW (3.3.7) (\*)     | `$FFTW_ROOT`, `$FFTW_ROOT_DIR` or `$FFTW_DIR` |
| HDF5 (1.10.1) (\*)    | `$HDF5_ROOT` or `$HDF5ROOT`                   |
| GSL (2.1) (\*)        | `$GSL_ROOT_DIR`                               |
| MPI                   | [See instructions][mpi-path] (\*\*)           |

(\*) Tested with the listed versions, higher versions probably work too.<br>
(\*\*) Especially if installing the dependencies, make sure that `mpicc` points to a MPI-compatible `C` compiler, and `mpic++` to a MPI-compatible `C++` compiler.

We suggest first trying to compile the code with the libraries already installed on the target machine or cluster.
If available, dependencies may be loaded with `module load ...` or `module load new ...`.
If `module load` is not available, but libraries are installed, set the above mentioned environment variables.
Installing all dependencies may require up to half an hour.

## Cluster-specific modules

#### Piz Daint:
```shell
module swap PrgEnv-cray PrgEnv-gnu
module load daint-gpu cray-python/3.6.1.1 cray-hdf5-parallel cray-fftw cray-petsc/3.8.4.0 cudatoolkit/9.2.148_3.19-6.0.7.1_2.1__g3d9acc8 CrayGNU/.18.08 GSL/2.5-CrayGNU-18.08 CMake/3.12.0
```
Then install accFFT
```shell
git clone https://github.com/novatig/accfft accfft
cd accfft
```
and follow the instructions in README.md


#### Euler:
First, load the following modules:
```shell
module load new modules gcc/6.3.0 open_mpi/2.1.1 fftw/3.3.4 binutils/2.25 gsl/1.16 hwloc/1.11.0
# For the single precision FFTW, load `fftw_sp/3.3.4` instead of `fftw/3.3.4`.
```
Then, manually install HDF5 using the provided script and add the required paths to your `.bashrc`:
```shell
./install_dependencies.sh --hdf5
echo "export HDF5_ROOT=$PWD/dependencies/build/hdf5-1.10.1-parallel/" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$HDF5_ROOT/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
```


## Installing dependencies manually

To install the missing dependencies, run the following code (from the repository root folder):
```bash
# Step 1: Install dependencies
./install_dependencies.sh --all

# Step 2: Append the export commands to ~/.bashrc or ~/.bash_profile (only for CMake)
./install_dependencies.sh --export >> ~/.bashrc
# or
./install_dependencies.sh --export >> ~/.bash_profile  # (Mac)

# Step 3:
source ~/.bashrc
# or
source ~/.bash_profile  # (Mac)

# Step 4: Try again
cd build
cmake ..
make
```

### Other options and further information

The `--all` flag installs all dependencies known to the script (FFTW, HDF5, GSL, as well as CMake itself).
If only some dependencies are missing, pass instead flags like `--cmake`, `--fftw` and othes.
Run `./install_dependencies.sh` to get the full list of available flags.

All dependencies are installed in the folder `dependencies/`.
Full installation takes 5-15 minutes, depending on the machine.
To specify the number of parallel jobs in the internal `make`, write
```
JOBS=10 ./install_dependencies.sh [flags]
```
The default number of jobs is equal to the number of physical cores.


## Advanced solver options (compile time)

The default options are sufficient for the average user.  However, advanced
users can customize the generated executable with the following options:

| Option                      | Default | Description                                                                                                                                                                                         |
|-----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CUP_UNBOUNDED_FFT`         | OFF     | This option enables an FFT based Poisson solver for isolated systems (see Hockney 1970).  Enabling this option will result in an improvement of accuracy at the cost of larger memory requirements. |
| `CUP_ASYNC_DUMP`            | ON      | This option enables asynchronous data dumps. If you run on a system with limited memory, this option can be disabled to reduce the memory footprint. Available only if MPI implementation is multithreaded (detected automatically). |
| `CUP_DUMP_SURFACE_BINARY`   | OFF     | Enabling this option dumps additional surface data for each obstacle in binary format.                                                                                                              |
| `CUP_SINGLE_PRECISION`      | OFF     | Run simulation in single precison.                                                                                                                                                                  |
| `CUP_HDF5_DOUBLE_PRECISION` | OFF     | Dump simulation snapshots in double precision.                                                                                                                                                      |
| `CUP_RK2`                   | OFF     | Enables a second order Runge-Kutta time integrator.                                                                                                                                                 |

These options can be enabled either on the command line with, e.g., `cmake
-DCUP_UNBOUNDED_FFT=ON` or with graphical tools such as `ccmake`.

## Compiling on Mac

The default compiler `clang` on Mac does not support OpenMP. It is therefore necessary either to install `clang` OpenMP extension or to install e.g. `g++` compiler. The following snippet shows how to compile with `g++-8`:
```bash
mkdir -p build
cd build
CC=gcc-8 CXX=g++-8 cmake ..
OMPI_CC=gcc-8 MPICH_CC=gcc-8 OMPI_CXX=g++-8 MPICH_CXX=g++-8 make
```

## Troubleshooting

If `cmake ..` keeps failing, delete the file `CMakeCache.txt` and try again.



# Configuring and running simulations

## Quick introduction

Perform the following steps:

1. Open `launch/config_form.html` in a web browser.

2. Configure the simulation.

3. Download the runscript (*Save*) or copy-paste it into a file.

4. Put the script into the `launch/` folder (recommended, otherwise see below).

5. `chmod +x scriptname.sh`

6. Run the script.

## Detailed description

For convenience, we have prepared a web form for customizing simulation settings, such as grid resolution, list of objects with their properties et cetera.
The form can be found in `launch/config_form.html` and can be opened with any modern browser.
Note that internet connection is required, as the web page uses external libraries for rendering and handling the form.

After setting up the system, or loading one of the presets, the generated runscript has to be run.

By default, the form assumes that the runscript will be located in the `launch/` folder. If not, adjust the *Repository root path* to the full path of the repository. If necessary, adjust also the *Results path*.
For debugging and reproducibility purposes, the runscript makes a copy of the executable and the current source code to the results path.
Additionally, the script copies itself, together with a compact representation of all form data, that can be later imported back (the last line of the generated runscript).


# Visualization using Paraview

This is a very quick overview on how to visualize results using [ParaView][paraview] (5.5.0).
For more complex visualization, please refer to the ParaView documentation.

### Visualizing shapes

1. Open `restart_*-chi.xmf`. Select *Xdmf Reader* option.

2. Press *Apply* in the *Properties* window (left).

3. If the snapshot is relatively small (< 100 MB), you can use the expensive *Volume* option. Change *Outline* to *Volume*.

4. If the snapshots are large, use *Slice* option. Go to *Filters* -> *Common* -> *Slice*. Change *Normal* to (0, 0, 1). Unselect *Show Plane*. Click *Apply*.

5. If the objects are not static, press the *Play* button to visualize the movement in time.

### Visualizing flow

1. Open `restart_*-vel.xmf`. Select *Xdmf Reader* option.

2. Press *Apply*.

3. Go to *Filters* -> *Common* -> *Slice*. Change *Normal* to (0, 0, 1). Unselect *Show Plane*. Click *Apply*.

4. In the toolbar, change *Solid Color* to *data*. Select *Magnitude* for the velocity magnitude, or *X*, *Y* or *Z* for respective components.

5. It may be necessary to adjust the color scale. Click *Rescale to Data Range* icon (possibly 2nd row, 4th icon).

6. Press *Play*.





[mpi-path]: https://stackoverflow.com/questions/43054602/custom-mpi-path-in-cmake-project
[paraview]: https://www.paraview.org/
