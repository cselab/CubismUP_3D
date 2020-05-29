//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_PoissonSolverACC_common_h
#define CubismUP_3D_PoissonSolverACC_common_h

#include "accfft_utils.h"
#include <string>

#ifndef CUP_SINGLE_PRECISION
  typedef double acc_c[2];
  #define MPIREAL MPI_DOUBLE
  #define cufftCmpT cufftDoubleComplex
  #define cufftPlanFWD CUFFT_D2Z
  #define cufftPlanBWD CUFFT_Z2D
  #define cufftExecFWD cufftExecD2Z
  #define cufftExecBWD cufftExecZ2D
  #define accfft_local_size accfft_local_size_dft_r2c_gpu
  #define accfft_plan_dft accfft_plan_dft_3d_r2c_gpu
  #define accfft_delplan accfft_destroy_plan_gpu
  #define accfft_clean accfft_cleanup_gpu
  #define accfft_exec_r2c accfft_execute_r2c_gpu
  #define accfft_exec_c2r accfft_execute_c2r_gpu
#else
  typedef float acc_c[2];
  #define MPIREAL MPI_FLOAT
  #define cufftCmpT cufftComplex
  #define cufftPlanFWD CUFFT_R2C
  #define cufftPlanBWD CUFFT_C2R
  #define cufftExecFWD cufftExecR2C
  #define cufftExecBWD cufftExecC2R
  #define accfft_local_size accfft_local_size_dft_r2c_gpuf
  #define accfft_plan_dft accfft_plan_dft_3d_r2c_gpuf
  #define accfft_delplan accfft_destroy_plan_gpu
  #define accfft_clean accfft_cleanup_gpuf
  #define accfft_exec_r2c accfft_execute_r2c_gpuf
  #define accfft_exec_c2r accfft_execute_c2r_gpuf
#endif

inline void printMemUse(const std::string where)
{
  size_t free_byte, total_byte ;
  cudaMemGetInfo( &free_byte, &total_byte ) ;
  double free_db=free_byte, total_db=total_byte, used_db=total_db-free_db;
  printf("%s: used = %f, free = %f MB, total = %f MB\n", where.c_str(),
    used_db/1024/1024, free_db/1024/1024, total_db/1024/1024); fflush(0);
}
#define CUDA_Check(code) do {  \
    if (code != cudaSuccess) { \
      printf("DONE DEAD func:%s file:%s:%d %s\n", __func__, \
      __FILE__,__LINE__, cudaGetErrorString(code)); \
    } \
  } while(0)

#endif // CubismUP_3D_PoissonSolverACC_common_h
