//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PoissonSolverACCPeriodic.h"
#include <cuda_runtime_api.h>
#include "PoissonSolverACC_common.h"
#include "accfft_common.h"
#ifndef CUP_SINGLE_PRECISION
  #include "accfft_gpu.h"
  typedef accfft_plan_gpu acc_plan;
#else
  #include "accfft_gpuf.h"
  typedef accfft_plan_gpuf acc_plan;
#endif

void _fourier_filter_gpu(
  acc_c*const __restrict__ data_hat, const size_t gsize[3],
  const int osize[3], const int ostart[3], const cubismup3d::Real h);

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

PoissonSolverPeriodic::PoissonSolverPeriodic(SimulationData & s) : PoissonSolver(s)
{
  if(s.nprocs > 1)
  {
    const size_t gz_hat = gsize[2] / 2 + 1;
    if (gsize[2]!=myN[2]) {
      printf("PoissonSolverPeriodic assumes grid is distrubuted in x and y.\n");
      abort();
    }
    int c_dims[2] = {
      static_cast<int>(gsize[0]/myN[0]), static_cast<int>(gsize[1]/myN[1])
    };
    assert(gsize[0]%myN[0]==0 && gsize[1]%myN[1]==0);
    accfft_create_comm(grid.getCartComm(), c_dims, &c_comm);
    testComm();
    int totN[3] = { (int)gsize[0], (int)gsize[1], (int)gsize[2] };

    alloc_max = accfft_local_size(totN, isize, istart, osize, ostart, c_comm);
    assert(alloc_max == isize[0] * isize[1] * 2*gz_hat * sizeof(Real));

    printf("[mpi rank %d] max:%lu isize:{%3d %3d %3d} osize:{%3d %3d %3d}\n"
      "istart:{%3d %3d %3d} ostart:{%3d %3d %3d} gsize:{%d %d %d}.\n", m_rank,
      alloc_max,isize[0],isize[1],isize[2],osize[0],osize[1],osize[2],istart[0],
      istart[1],istart[2],ostart[0],ostart[1],ostart[2],totN[0],totN[1],totN[2]);
    fflush(0);

    if(isize[0]!=(int)myN[0] || isize[1]!=(int)myN[1] || isize[2]!=(int)myN[2]) {
      printf("PoissonSolverPeriodic: something wrong in isize\n");
      abort();
    }

    cudaMalloc((void**) &phi_hat, alloc_max);

    acc_plan* P = accfft_plan_dft(totN, phi_hat,phi_hat, c_comm,ACCFFT_MEASURE);
    plan = (void*) P;

    data = (Real*) malloc(myN[0] * myN[1] * 2*gz_hat * sizeof(Real));
    data_size = (size_t) myN[0] * (size_t) myN[1] * (size_t) 2*gz_hat;
    stridez = 1; // fast
    stridey = 2*gz_hat;
    stridex = myN[1] * 2*gz_hat; // slow
  }
  else
  {
    // for cuFFT we use x as fast index instead of z:
    const size_t gx_hat = gsize[0] / 2 + 1;
    osize[0] = myN[2]; isize[0] = myN[2];
    osize[1] = myN[1]; isize[1] = myN[1];
    osize[2] = myN[0]; isize[2] = myN[0];
    ostart[0] = 0; istart[0] = 0;
    ostart[1] = 0; istart[1] = 0;
    ostart[2] = 0; istart[2] = 0;
    alloc_max = 0;
    printf("[mpi rank %d] istart:{%3d %3d %3d} ostart:{%3d %3d %3d}\n"
      "isize:{%3d %3d %3d} osize:{%3d %3d %3d} gsize:{%lu %lu %lu}.\n", m_rank,
      istart[0],istart[1],istart[2], ostart[0],ostart[1],ostart[2], isize[0],
      isize[1],isize[2],osize[0],osize[1],osize[2],gsize[0],gsize[1],gsize[2]);
    fflush(0);
    cufftPlan3d(&cufft_fwd, myN[2], myN[1], myN[0], cufftPlanFWD);
    cufftPlan3d(&cufft_bwd, myN[2], myN[1], myN[0], cufftPlanBWD);
    cudaMalloc((void**) &phi_hat, myN[2] * myN[1] * gx_hat * sizeof(cufftCmpT));

    data_size = (size_t) myN[2] * (size_t) myN[1] * (size_t) 2*gx_hat;
    data = (Real*) malloc(data_size * sizeof(Real));
    stridez = myN[1] * 2*gx_hat; // slow
    stridey = 2*gx_hat;
    stridex = 1; // fast
  }
}

void PoissonSolverPeriodic::solve()
{
  if(sim.nprocs > 1) solve_multiNode();
  else solve_singleNode();
}

void PoissonSolverPeriodic::solve_multiNode()
{
  sim.startProfiler("ACCDFT cub2rhs");
  _cub2fftw();
  sim.stopProfiler();

  sim.startProfiler("ACCDFT cpu2gpu");
  cudaMemcpy(phi_hat, data, alloc_max, cudaMemcpyHostToDevice);
  sim.stopProfiler();

  // Perform forward FFT
  sim.startProfiler("ACCDFT r2c");
  accfft_exec_r2c((acc_plan*)plan, phi_hat, (acc_c*)phi_hat);
  sim.stopProfiler();

  // Spectral solve
  sim.startProfiler("ACCDFT solve");
  _fourier_filter_gpu((acc_c*)phi_hat, gsize, osize, ostart, h);
  sim.stopProfiler();

  // Perform backward FFT
  sim.startProfiler("ACCDFT c2r");
  accfft_exec_c2r((acc_plan*)plan, (acc_c*)phi_hat, phi_hat);
  sim.stopProfiler();

  sim.startProfiler("ACCDFT gpu2cpu");
  cudaMemcpy(data, phi_hat, alloc_max, cudaMemcpyDeviceToHost);
  sim.stopProfiler();
}

void PoissonSolverPeriodic::solve_singleNode()
{
  sim.startProfiler("ACCDFT cub2rhs");
  _cub2fftw();
  sim.stopProfiler();

  sim.startProfiler("ACCDFT cpu2gpu");
  cudaMemcpy(phi_hat, data, data_size * sizeof(Real), cudaMemcpyHostToDevice);
  sim.stopProfiler();

  // Perform forward FFT
  sim.startProfiler("ACCDFT r2c");
  cufftExecFWD(cufft_fwd, phi_hat, (cufftCmpT*) phi_hat);
  sim.stopProfiler();

  // Spectral solve
  sim.startProfiler("ACCDFT solve");
  // for cuFFT we use x as fast index instead of z:
  const size_t gsize_T[3] = {gsize[2], gsize[1], gsize[0]};
  _fourier_filter_gpu((acc_c*)phi_hat, gsize_T, osize, ostart, h);
  //_fourier_filter_gpu_transp((cufftCmpT*)phi_hat, gsize, osize, ostart, h);
  sim.stopProfiler();

  // Perform backward FFT
  sim.startProfiler("ACCDFT c2r");
  cufftExecBWD(cufft_bwd, (cufftCmpT*) phi_hat, phi_hat);
  sim.stopProfiler();

  sim.startProfiler("ACCDFT gpu2cpu");
  cudaMemcpy(data, phi_hat, data_size * sizeof(Real), cudaMemcpyDeviceToHost);
  sim.stopProfiler();
}

PoissonSolverPeriodic::~PoissonSolverPeriodic()
{
  free(data);
  cudaFree(phi_hat);
  if(sim.nprocs > 1) {
    accfft_destroy_plan_gpu((acc_plan*)plan);
    accfft_clean();
    MPI_Comm_free(&c_comm);
  } else {
    cufftDestroy(cufft_fwd);
    cufftDestroy(cufft_bwd);
  }
}

void PoissonSolverPeriodic::testComm()
{
  int accfft_rank, accfft_size, cubism_rank, cubism_size;
  MPI_Comm_rank( c_comm, &accfft_rank);
  MPI_Comm_size( c_comm, &accfft_size);
  MPI_Comm_rank( grid.getCartComm(), &cubism_rank);
  MPI_Comm_size( grid.getCartComm(), &cubism_size);
  int accfft_left, accfft_right, cubism_left, cubism_right;
  MPI_Cart_shift(c_comm, 0, 1, &accfft_left,   &accfft_right);
  MPI_Cart_shift(grid.getCartComm(), 0, 1, &cubism_left,   &cubism_right);
  int accfft_bottom, accfft_top, cubism_bottom, cubism_top;
  MPI_Cart_shift(c_comm, 1, 1, &accfft_bottom, &accfft_top);
  MPI_Cart_shift(grid.getCartComm(), 1, 1, &cubism_bottom, &cubism_top);
  //int accfft_front, accfft_back, cubism_front, cubism_back;
  //MPI_Cart_shift(c_comm, 2, 1, &accfft_front,  &accfft_back);
  //MPI_Cart_shift(grid.getCartComm(), 2, 1, &cubism_front,  &cubism_back);
  //note: accfft comm is not periodic and 2d, cubism is periodic adn 3d, rest must be the same
  #if 0
  if( accfft_left   != cubism_left   || accfft_right != cubism_right  ||
      accfft_bottom != cubism_bottom || accfft_top   != cubism_top    ||
      accfft_rank   != cubism_rank   || accfft_size  != cubism_size   //||
      //accfft_front  != cubism_front  || accfft_back  != cubism_back
    )
  #else
  if( ( accfft_left  !=MPI_PROC_NULL && accfft_left  !=cubism_left   ) ||
      ( accfft_right !=MPI_PROC_NULL && accfft_right !=cubism_right  ) ||
      ( accfft_bottom!=MPI_PROC_NULL && accfft_bottom!=cubism_bottom ) ||
      ( accfft_top   !=MPI_PROC_NULL && accfft_top   !=cubism_top    ) ||
      ( accfft_rank  !=cubism_rank   || accfft_size  !=cubism_size   ) //||
      //( accfft_front !=MPI_PROC_NULL && accfft_front !=cubism_front  ) ||
      //( accfft_back  !=MPI_PROC_NULL && accfft_back  !=cubism_back   )
     )
  #endif
  {
    printf("AccFFT communicator does not match the one from Cubism. Aborting.\n");
    fflush(0);
    MPI_Abort(grid.getCartComm(), MPI_ERR_OTHER);
  }
}

CubismUP_3D_NAMESPACE_END
