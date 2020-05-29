# NOTE: Update `install_dependencies.sh` if updating the minimum required version.
cmake_minimum_required(VERSION 3.2)
project(CubismUP_3D)

# Choose Release mode as default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(ROOT_FOLDER ${CMAKE_CURRENT_LIST_DIR})


# https://stackoverflow.com/a/31832359
function(ARRAY_JOIN OUTPUT GLUE)
    set(_TMP_RESULT "")
    set(_GLUE "") # effective glue is empty at the beginning
    foreach(arg ${ARGN})
        set(_TMP_RESULT "${_TMP_RESULT}${_GLUE}${arg}")
        set(_GLUE "${GLUE}")
    endforeach()
    set(${OUTPUT} "${_TMP_RESULT}" PARENT_SCOPE)
endfunction()


# Options
set(EXE "cubismup3d_simulation")
set(BLOCK_SIZE "16" CACHE String "Number of grid points in a block, per dimension")

option(COMPILE_EXE "Compile executable" OFF)
option(COMPILE_STATIC_LIB "Compile static library" ON)
option(COMPILE_PY_SO "Compile Python bindings" OFF)

# Compile-time settings. Stored in a configuration file CubismUP3DMacros.h and imported by Definitions.h.
option(CUP_ASYNC_DUMP "Use asynchronous data dumps (requires additional memory)" ON)
option(CUP_DUMP_SURFACE_BINARY "Dump binary surface data for each obstacle" OFF)
option(CUP_HDF5_DOUBLE_PRECISION "Dump HDF5 data in double precision" OFF)
option(CUP_SINGLE_PRECISION "Compute in single precision" OFF)  # Because cmake/FindFFTW.cmake handles now only double precision.
option(CUP_VERBOSE "Be verbose" OFF)

# Pybind11.
# Not using `python3-config` because it doesn't seem to work with `pip install --user`.
execute_process(
    COMMAND python3 -m pybind11 --includes
    OUTPUT_VARIABLE PYBIND11_INCLUDES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode1)
execute_process(
    #COMMAND python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX')[:-3])"
    COMMAND python3-config --extension-suffix
    OUTPUT_VARIABLE PY_SO_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode2)
execute_process(
    COMMAND python3-config --ldflags
    OUTPUT_VARIABLE PYTHON_LIBRARIES
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE retcode3)
separate_arguments(PYBIND11_INCLUDES)
set(PYBIND11_FLAGS -shared -fPIC ${PYBIND11_INCLUDES})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(PYBIND11_FLAGS ${PYBIND11_FLAGS} -undefined dynamic_lookup)
endif()

if (COMPILE_PY_SO)
    if (NOT "${retcode1}" STREQUAL "0")
        message(FATAL_ERROR "pybind11 not installed, cannot compile Python bindings!")
    endif()
    if (NOT "${retcode2}" STREQUAL "0")
        message(FATAL_ERROR "Cannot get which Python .so suffix to use, cannot compile bindings!")
    endif()
    if (NOT "${retcode3}" STREQUAL "0")
        message(FATAL_ERROR "Command `python3-config --ldflags` failed, cannot compile bindings!")
    endif()
endif()


# MPI - must be before modifying CMAKE_SHARED_LIBRARY_SUFFIX (?).
find_package(MPI REQUIRED)
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# Check if MPI is thread safe.
execute_process(COMMAND ${CMAKE_CXX_COMPILER} ${ROOT_FOLDER}/cmake/check_mpi_thread_safety.cpp -o check_mpi_thread_safety
                RESULT_VARIABLE _IS_COMPILED)
if (NOT _IS_COMPILED STREQUAL "0")
    message(FATAL_ERROR "Cannot compile a simple MPI program to determine whether MPI is thread safe.")
endif()
execute_process(COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ./check_mpi_thread_safety
                ERROR_QUIET
                OUTPUT_QUIET
                RESULT_VARIABLE CUP_IS_MPI_THREAD_UNSAFE)
if (${CUP_IS_MPI_THREAD_UNSAFE})
    message("Note: MPI not thread safe, fallback to synchronous HDF5 dumping!")
endif()


# Source files and target.
set(STATIC_LIB "cubismup3d")
set(STATIC_LIB_MERGED "cubismup3d_merged")  # Merged with 3rd party libraries.
set(PY_SO "cubismup3d${PY_SO_SUFFIX}")

set(CMAKE_SHARED_LIBRARY_PREFIX "")      # No "lib" prefix for shared.
set(CMAKE_SHARED_LIBRARY_SUFFIX "")      # We already specify it manually.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ".")

include_directories(${ROOT_FOLDER}/Cubism/include/)
set(COMMON_SOURCES           # Common for app and lib.
    ${ROOT_FOLDER}/Cubism/src/ArgumentParser.cpp  # Temporary solution for Cubism .cpp files.
    ${ROOT_FOLDER}/source/utils/BufferedLogger.cpp

    ${ROOT_FOLDER}/source/obstacles/CarlingFish.cpp
    ${ROOT_FOLDER}/source/obstacles/Cylinder.cpp
    # ${ROOT_FOLDER}/source/obstacles/ElasticFish.cpp
    ${ROOT_FOLDER}/source/obstacles/Ellipsoid.cpp
    ${ROOT_FOLDER}/source/obstacles/ExternalObstacle.cpp
    ${ROOT_FOLDER}/source/obstacles/Fish.cpp
    ${ROOT_FOLDER}/source/obstacles/FishLibrary.cpp
    ${ROOT_FOLDER}/source/obstacles/FishShapes.cpp
    ${ROOT_FOLDER}/source/obstacles/Naca.cpp
    ${ROOT_FOLDER}/source/obstacles/Obstacle.cpp
    ${ROOT_FOLDER}/source/obstacles/ObstacleFactory.cpp
    ${ROOT_FOLDER}/source/obstacles/ObstacleVector.cpp
    ${ROOT_FOLDER}/source/obstacles/Plate.cpp
    ${ROOT_FOLDER}/source/obstacles/Sphere.cpp
    ${ROOT_FOLDER}/source/obstacles/StefanFish.cpp

    ${ROOT_FOLDER}/source/operators/AdvectionDiffusion.cpp
    ${ROOT_FOLDER}/source/operators/Analysis.cpp
    ${ROOT_FOLDER}/source/operators/Checkpoint.cpp
    ${ROOT_FOLDER}/source/operators/ComputeDissipation.cpp
    ${ROOT_FOLDER}/source/operators/ExternalForcing.cpp
    ${ROOT_FOLDER}/source/operators/FadeOut.cpp
    ${ROOT_FOLDER}/source/operators/FixedMassFlux_nonUniform.cpp
    ${ROOT_FOLDER}/source/operators/FluidSolidForces.cpp
    ${ROOT_FOLDER}/source/operators/InitialConditions.cpp
    ${ROOT_FOLDER}/source/operators/IterativePressureNonUniform.cpp
    ${ROOT_FOLDER}/source/operators/IterativePressurePenalization.cpp
    ${ROOT_FOLDER}/source/operators/ObstaclesCreate.cpp
    ${ROOT_FOLDER}/source/operators/ObstaclesUpdate.cpp
    ${ROOT_FOLDER}/source/operators/Penalization.cpp
    ${ROOT_FOLDER}/source/operators/PressureProjection.cpp
    ${ROOT_FOLDER}/source/operators/PressureRHS.cpp
    ${ROOT_FOLDER}/source/operators/SGS.cpp
    ${ROOT_FOLDER}/source/poisson/PoissonSolver.cpp
    ${ROOT_FOLDER}/source/poisson/PoissonSolverMixed.cpp
    ${ROOT_FOLDER}/source/poisson/PoissonSolverPeriodic.cpp
    ${ROOT_FOLDER}/source/poisson/PoissonSolverUnbounded.cpp
    ${ROOT_FOLDER}/source/spectralOperators/SpectralAnalysis.cpp
    ${ROOT_FOLDER}/source/spectralOperators/SpectralForcing.cpp
    ${ROOT_FOLDER}/source/spectralOperators/SpectralIcGenerator.cpp
    ${ROOT_FOLDER}/source/spectralOperators/SpectralManip.cpp
    ${ROOT_FOLDER}/source/spectralOperators/SpectralManipFFTW.cpp

    ${ROOT_FOLDER}/source/Simulation.cpp
    ${ROOT_FOLDER}/source/SimulationData.cpp
)

# GCC flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-unused-parameter -Wno-maybe-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmax-errors=3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -march=native -fstrict-aliasing -ffast-math -falign-functions=32")
set(CMAKE_CXX_EXTENSIONS OFF)  # -std=c++17, not -std=gnu++17!
if (NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

# Targets.
if (COMPILE_EXE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ROOT_FOLDER}/bin)
    add_executable(${EXE} ${COMMON_SOURCES} ${ROOT_FOLDER}/source/main.cpp)
endif()
if (COMPILE_STATIC_LIB)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${ROOT_FOLDER}/build/lib)
    add_library(${STATIC_LIB}
                ${COMMON_SOURCES}
                ${ROOT_FOLDER}/source/cubism_main.cpp)
endif()
if (COMPILE_PY_SO)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${ROOT_FOLDER}/python)
    add_library(${PY_SO} SHARED
                ${COMMON_SOURCES}
                ${ROOT_FOLDER}/source/bindings/Obstacles.cpp
                ${ROOT_FOLDER}/source/bindings/Simulation.cpp)
    target_compile_options(${PY_SO} PRIVATE ${PYBIND11_FLAGS})
    target_link_libraries(${PY_SO} ${PYTHON_LIBRARIES})
    # https://github.com/conda-forge/hdf5-feedstock/issues/58#issuecomment-310865909
    target_compile_definitions(${PY_SO} PRIVATE "-DH5_BUILT_AS_DYNAMIC_LIB")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# OpenMP support
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

# HDF5
if (NOT HDF5_ROOT AND EXISTS ${ROOT_FOLDER}/dependencies/build/hdf5-1.10.1)
    set(HDF5_ROOT ${ROOT_FOLDER}/dependencies/build/hdf5-1.10.1)
endif()
set(HDF5_PREFER_PARALLEL ON)
find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

# FFTW
# TODO: Single precision
if (NOT FTTW_ROOT AND NOT DEFINED ENV{FFTWDIR} AND EXISTS ${ROOT_FOLDER}/dependencies/build/fftw-3.3.7)
    set(FFTW_ROOT ${ROOT_FOLDER}/dependencies/build/fftw-3.3.7)
endif()
set(FFTW_USE_STATIC_LIBS 1)
find_package(FFTW REQUIRED)
include_directories(${FFTW_INCLUDES})

# GSL
if (NOT GSL_ROOT_DIR AND NOT DEFINED ENV{GSL_ROOT_DIR} AND EXISTS ${ROOT_FOLDER}/dependencies/build/gsl-2.1)
    set(GSL_ROOT_DIR ${ROOT_FOLDER}/dependencies/build/gsl-2.1)
endif()
find_package(GSL REQUIRED)
include_directories(${GSL_INCLUDE_DIRS})

if (COMPILE_EXE)
    target_link_libraries(${EXE} ${HDF5_LIBRARIES})
    target_link_libraries(${EXE} ${FFTW_LIBRARIES})
    target_link_libraries(${EXE} ${GSL_LIBRARIES})
endif()
if (COMPILE_STATIC_LIB)
    target_link_libraries(${STATIC_LIB} ${HDF5_LIBRARIES})
    target_link_libraries(${STATIC_LIB} ${FFTW_LIBRARIES})
    target_link_libraries(${STATIC_LIB} ${GSL_LIBRARIES})

    # For applications that use CubismUP3D as a library, prepare a list of all
    # link dependencies and store them in a file as a space-separated list.
    get_target_property(_LINK_DEPENDENCIES ${STATIC_LIB} LINK_LIBRARIES)
    array_join(_LINK_DEPENDENCIES " " ${_LINK_DEPENDENCIES})

    # `configure_file` will touch the file only if the new content is different
    # than the existing one (if any), which makes it Makefile-friendly.
    configure_file(${ROOT_FOLDER}/cmake/CubismUP3DDependencies.template.txt
                   ${ROOT_FOLDER}/build/CubismUP3DDependencies.txt)
endif()
if (COMPILE_PY_SO)
    target_link_libraries(${PY_SO} ${HDF5_LIBRARIES})
    target_link_libraries(${PY_SO} ${FFTW_LIBRARIES})
    target_link_libraries(${PY_SO} ${GSL_LIBRARIES})
endif()

if (COMPILE_STATIC_LIB)
    enable_testing()
    add_subdirectory(tests)
endif()

# Generate macro file with current compilation settings. This file is generated
# at configuration time (`cmake ..`), so it should be available for compilation # immediately.
# https://cmake.org/cmake/help/latest/command/configure_file.html
configure_file(${ROOT_FOLDER}/cmake/CubismUP3DMacros.template.h ${ROOT_FOLDER}/build/include/CubismUP3DMacros.h)
