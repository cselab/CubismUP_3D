//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "SpectralManipACC.h"
#include <cuda_runtime_api.h>
#include "../poisson/PoissonSolverACC_common.h"
#include "accfft_common.h"
#ifndef CUP_SINGLE_PRECISION
  #include "accfft_gpu.h"
  typedef accfft_plan_gpu acc_plan;
#else
  #include "accfft_gpuf.h"
  typedef accfft_plan_gpuf acc_plan;
#endif

#ifndef CUP_SINGLE_PRECISION
#define MPIREAL MPI_DOUBLE
#else
#define MPIREAL MPI_FLOAT
#endif /* CUP_SINGLE_PRECISION */

using treal = cubismup3d::Real;
void _compute_HIT_analysis(
  acc_c * const Uhat, acc_c * const Vhat, acc_c * const What,
  const size_t gsize[3], const int osize[3], const int ostart[3],
  const treal h, treal & tke, treal & eps, treal & lInt, treal & tkeFiltered,
  treal * const eSpectrum, const size_t nBins, const treal nyquist
);

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

struct myCUDAstreams
{
  cudaStream_t u;
  cudaStream_t v;
  cudaStream_t w;
  myCUDAstreams()
  {
    cudaStreamCreate ( & u );
    cudaStreamCreate ( & v );
    cudaStreamCreate ( & w );
  }
  ~myCUDAstreams()
  {
    // how to deallocate?
  }
};

void SpectralManipACC::_compute_forcing()
{
  sim.stopProfiler();
  sim.startProfiler("ACCForce ansys");

  const double h = sim.uniformH();
  const size_t nBins = stats.nBin;
  const Real nyquist = stats.nyquist;
  Real tke = 0, eps = 0, lIntegral = 0, tkeFiltered = 0;
  Real * const E_msr = stats.E_msr;
  //std::vector<Real> Eback(E_msr, E_msr + nBins);
  memset(E_msr, 0, nBins * sizeof(Real));

  // kernel
  size_t gsize_[3] = {gsize[0], gsize[1], gsize[2]};
  // for cuFFT we use x as fast index instead of z:
  if(sim.nprocs < 2) { gsize_[0] = gsize[2]; gsize_[2] = gsize[0]; }
  _compute_HIT_analysis( (acc_c*) gpu_u, (acc_c*) gpu_v, (acc_c*) gpu_w, gsize_,
    osize, ostart, h, tke, eps, lIntegral, tkeFiltered, E_msr, nBins, nyquist);

  MPI_Allreduce(MPI_IN_PLACE, &tke, 1, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, &eps, 1, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, &lIntegral, 1, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, &tkeFiltered, 1, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, E_msr, nBins, MPIREAL, MPI_SUM, m_comm);

  const Real normalization = 1.0 / pow2(normalizeFFT);
  for (size_t binID = 0; binID < nBins; binID++) E_msr[binID] *= normalization;

  stats.tke = tke * normalization;
  stats.l_integral = lIntegral / tke;
  stats.tke_filtered = tkeFiltered * normalization;
  stats.dissip_visc = 2 * eps * sim.nu * normalization / pow2(2*h);
  //stats.dissip_visc = eps * 2 * sim.nu * normalization;
  //std::cout << "step " << sim.step <<" E diff:";
  //for (size_t i = 0; i < nBins; i++) {
  //  assert(E_msr[i] >= 0 && Eback[i] >= 0);
  //  std::cout<<' '<<(E_msr[i]-Eback[i])/std::max({E_msr[i], Eback[i], 1e-16});
  //}
  //std::cout << '\n';
  sim.stopProfiler();
  sim.startProfiler("SpectralForcing");
}

void SpectralManipACC::_compute_IC(const std::vector<Real> &K,
                                   const std::vector<Real> &E)
{
  printf("ABORT: SpectralManipACC does not support _compute_IC\n");
  fflush(0); abort();
}

SpectralManipACC::SpectralManipACC(SimulationData&s): SpectralManip(s),
streams(new myCUDAstreams())
{
  if(s.nprocs > 1)
  {
    const size_t nz_hat = gsize[2]/2+1;
    if (gsize[2]!=myN[2]) {
      printf("SpectralManipACC assumes grid is distrubuted in x and y.\n");
      abort();
    }
    int c_dims[2] = {
      static_cast<int>(gsize[0]/myN[0]), static_cast<int>(gsize[1]/myN[1])
    };
    assert(gsize[0]%myN[0]==0 && gsize[1]%myN[1]==0);
    accfft_create_comm(grid.getCartComm(), c_dims, &acc_comm);
    int totN[3] = { (int)gsize[0], (int)gsize[1], (int)gsize[2] };

    alloc_max = accfft_local_size(totN, isize, istart, osize, ostart, acc_comm);
    assert(alloc_max == isize[0] * isize[1] * 2*nz_hat * sizeof(Real));

    if(isize[0]!=(int)myN[0] || isize[1]!=(int)myN[1] || isize[2]!=(int)myN[2])
    {
      printf("PoissonSolverPeriodic: something wrong in isize\n");
      abort();
    }
    cudaMalloc((void**) & gpu_u, alloc_max);
    cudaMalloc((void**) & gpu_v, alloc_max);
    cudaMalloc((void**) & gpu_w, alloc_max);
    acc_plan* P = accfft_plan_dft(totN, gpu_u, gpu_u, acc_comm, ACCFFT_MEASURE);
    plan = (void*) P;
    data_size = (size_t) myN[0] * (size_t) myN[1] * (size_t) 2*nz_hat;
    stridez = 1; // fast
    stridey = 2*nz_hat;
    stridex = myN[1] * 2*nz_hat; // slow
  }
  else
  {
    const size_t nx_hat = gsize[0]/2+1;
    cufftPlan3d(&cufft_fwd, myN[2], myN[1], myN[0], cufftPlanFWD);
    cufftPlan3d(&cufft_bwd, myN[2], myN[1], myN[0], cufftPlanBWD);
    cudaMalloc((void**) & gpu_u, myN[2]*myN[1]*nx_hat * sizeof(cufftCmpT) );
    cudaMalloc((void**) & gpu_v, myN[2]*myN[1]*nx_hat * sizeof(cufftCmpT) );
    cudaMalloc((void**) & gpu_w, myN[2]*myN[1]*nx_hat * sizeof(cufftCmpT) );
    osize[0] = myN[2]; isize[0] = myN[2];
    osize[1] = myN[1]; isize[1] = myN[1];
    osize[2] = myN[0]; isize[2] = myN[0];
    ostart[0] = 0; istart[0] = 0;
    ostart[1] = 0; istart[1] = 0;
    ostart[2] = 0; istart[2] = 0;
    data_size = (size_t) myN[2] * (size_t) myN[1] * (size_t) 2*nx_hat;
    stridez = myN[1] * 2*nx_hat; // fast
    stridey = 2*nx_hat;
    stridex = 1; // slow
  }
  data_u = (Real*) malloc(data_size * sizeof(Real));
  data_v = (Real*) malloc(data_size * sizeof(Real));
  data_w = (Real*) malloc(data_size * sizeof(Real));
  #ifdef ENERGY_FLUX_SPECTRUM
  data_j = (Real*) malloc(data_size * sizeof(Real));
  #endif
}

void SpectralManipACC::prepareFwd()
{
}

void SpectralManipACC::prepareBwd()
{
}

void SpectralManipACC::runFwd() const
{
  sim.stopProfiler();
  sim.startProfiler("ACCForce fwd");
  const size_t bufSize = data_size * sizeof(Real);
  cudaMemcpyAsync(gpu_u, data_u, bufSize, cudaMemcpyHostToDevice, streams->u);
  cudaMemcpyAsync(gpu_v, data_v, bufSize, cudaMemcpyHostToDevice, streams->v);
  cudaMemcpyAsync(gpu_w, data_w, bufSize, cudaMemcpyHostToDevice, streams->w);
  //cudaMemcpy(gpu_u, data_u, bufSize, cudaMemcpyHostToDevice);
  //cudaMemcpy(gpu_v, data_v, bufSize, cudaMemcpyHostToDevice);
  //cudaMemcpy(gpu_w, data_w, bufSize, cudaMemcpyHostToDevice);
  if(sim.nprocs > 1)
  {
  cudaStreamSynchronize ( streams->u );
  accfft_exec_r2c((acc_plan*)plan, gpu_u, (acc_c*)gpu_u);
  cudaStreamSynchronize ( streams->v );
  accfft_exec_r2c((acc_plan*)plan, gpu_v, (acc_c*)gpu_v);
  cudaStreamSynchronize ( streams->w );
  accfft_exec_r2c((acc_plan*)plan, gpu_w, (acc_c*)gpu_w);
  }
  else
  {
  cudaStreamSynchronize ( streams->u );
  cufftExecFWD(cufft_fwd, gpu_u, (cufftCmpT*)gpu_u);
  cudaStreamSynchronize ( streams->v );
  cufftExecFWD(cufft_fwd, gpu_v, (cufftCmpT*)gpu_v);
  cudaStreamSynchronize ( streams->w );
  cufftExecFWD(cufft_fwd, gpu_w, (cufftCmpT*)gpu_w);
  }

  sim.stopProfiler();
  sim.startProfiler("SpectralForcing");
}

void SpectralManipACC::runBwd() const
{
  sim.stopProfiler();
  sim.startProfiler("ACCForce bwd");
  const size_t bufSize = data_size * sizeof(Real);

  if(sim.nprocs > 1)
  {
  accfft_exec_c2r((acc_plan*)plan, (acc_c*)gpu_u, gpu_u);
  cudaMemcpyAsync(data_u, gpu_u, alloc_max, cudaMemcpyDeviceToHost, streams->u);
  accfft_exec_c2r((acc_plan*)plan, (acc_c*)gpu_v, gpu_v);
  cudaMemcpyAsync(data_v, gpu_v, alloc_max, cudaMemcpyDeviceToHost, streams->v);
  accfft_exec_c2r((acc_plan*)plan, (acc_c*)gpu_w, gpu_w);
  cudaMemcpyAsync(data_w, gpu_w, alloc_max, cudaMemcpyDeviceToHost, streams->w);
  }
  else
  {
  cufftExecBWD(cufft_bwd, (cufftCmpT*)gpu_u, gpu_u);
  cudaMemcpyAsync(data_u, gpu_u, bufSize, cudaMemcpyDeviceToHost, streams->u);
  cufftExecBWD(cufft_bwd, (cufftCmpT*)gpu_v, gpu_v);
  cudaMemcpyAsync(data_v, gpu_v, bufSize, cudaMemcpyDeviceToHost, streams->v);
  cufftExecBWD(cufft_bwd, (cufftCmpT*)gpu_w, gpu_w);
  cudaMemcpyAsync(data_w, gpu_w, bufSize, cudaMemcpyDeviceToHost, streams->w);
  }
  cudaStreamSynchronize ( streams->u );
  cudaStreamSynchronize ( streams->v );
  cudaStreamSynchronize ( streams->w );
  //cudaMemcpy(data_u, gpu_u, bufSize, cudaMemcpyDeviceToHost);
  //cudaMemcpy(data_v, gpu_v, bufSize, cudaMemcpyDeviceToHost);
  //cudaMemcpy(data_w, gpu_w, bufSize, cudaMemcpyDeviceToHost);

  sim.stopProfiler();
  sim.startProfiler("SpectralForcing");
}

SpectralManipACC::~SpectralManipACC()
{
  free(gpu_u);
  free(gpu_v);
  free(gpu_w);
  //cudaFree(rho_gpu);
  cudaFree(gpu_u);
  cudaFree(gpu_v);
  cudaFree(gpu_w);
  if(sim.nprocs > 1) {
    accfft_destroy_plan_gpu((acc_plan*)plan);
    accfft_clean();
    MPI_Comm_free(&acc_comm);
  } else {
    cufftDestroy(cufft_fwd);
    cufftDestroy(cufft_bwd);
  }
  delete streams;
}

CubismUP_3D_NAMESPACE_END
#undef MPIREAL

