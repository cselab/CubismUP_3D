//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PoissonSolverPeriodic.h"
#include "PoissonSolver_common.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

void PoissonSolverPeriodic::_solve()
{
  fft_c *const in_out = (fft_c *) data;
  // RHS comes into this function premultiplied by h^3 (as in FMM)
  #if 1 // grid-consistent
    // Solution has to be normalized (1/N^3) and multiplied by Laplace op finite
    // diffs coeff. We use finite diffs consistent with press proj, therefore
    // +/- 2h, and Poisson coef is (2h)^2. Due to RHS premultiplied by h^3:
    const Real norm_factor = 4/(gsize[0]*h*gsize[1]*gsize[2]);
    const long nKx = (long)gsize[0], nKy = (long)gsize[1], nKz = (long)gsize[2];
    const Real wFacX = 4*M_PI / nKx, wFacY = 4*M_PI / nKy, wFacZ = 4*M_PI / nKz;
    #pragma omp parallel for schedule(static)
    for(long j = 0; j<static_cast<long>(local_n1); ++j)
    for(long i = 0; i<static_cast<long>(gsize[0]); ++i)
    for(long k = 0; k<static_cast<long>(nz_hat);   ++k)
    {
      const size_t linidx = (j*gsize[0] +i)*nz_hat + k;
      const long l = local_1_start + j; //memory index plus shift due to decomp
      const Real D = std::cos(wFacX*i) +std::cos(wFacY*l) +std::cos(wFacZ*k);
      const Real solutionFactor = norm_factor / (2*D - 6);
      in_out[linidx][0] *= solutionFactor;
      in_out[linidx][1] *= solutionFactor;
    }

    // there are 8 modes that are undefined with this solution
    const long lastI = nKx/2, lastJ = nKy/2 - local_1_start, lastK = nKz/2;
    if (local_1_start == 0) {
      const size_t idWSF = (0*gsize[0] +0)*nz_hat + 0;
      const size_t idWSB = (0*gsize[0] +0)*nz_hat + lastK;
      const size_t idESF = (0*gsize[0] +lastI)*nz_hat + 0;
      const size_t idESB = (0*gsize[0] +lastI)*nz_hat + lastK;
      in_out[idWSF][0] = 0; in_out[idWSF][1] = 0;
      in_out[idWSB][0] = 0; in_out[idWSB][1] = 0;
      in_out[idESF][0] = 0; in_out[idESF][1] = 0;
      in_out[idESB][0] = 0; in_out[idESB][1] = 0;
    }
    if(local_1_start <= nKy/2 && local_1_start+local_n1 > nKy/2) {
      assert(lastJ < (long) local_n1);
      const size_t idWNF = (lastJ*gsize[0] +0)*nz_hat + 0;
      const size_t idWNB = (lastJ*gsize[0] +0)*nz_hat + lastK;
      const size_t idENF = (lastJ*gsize[0] +lastI)*nz_hat + 0;
      const size_t idENB = (lastJ*gsize[0] +lastI)*nz_hat + lastK;
      in_out[idWNF][0] = 0; in_out[idWNF][1] = 0;
      in_out[idWNB][0] = 0; in_out[idWNB][1] = 0;
      in_out[idENF][0] = 0; in_out[idENF][1] = 0;
      in_out[idENB][0] = 0; in_out[idENB][1] = 0;
    }

  #else // spectral
    // Solution has to be normalized by (1/N^3) and we take out h^3 factor:
    const Real norm_factor = 1/(gsize[0]*h*gsize[1]*h*gsize[2]*h);
    const long nKx = (long)gsize[0], nKy = (long)gsize[1], nKz = (long)gsize[2];
    const Real wFacX=2*M_PI/(nKx*h), wFacY=2*M_PI/(nKy*h), wFacZ=2*M_PI/(nKz*h);
    #pragma omp parallel for schedule(static)
    for(long j = 0; j<static_cast<long>(local_n1); ++j)
    for(long i = 0; i<static_cast<long>(gsize[0]); ++i)
    for(long k = 0; k<static_cast<long>(nz_hat);   ++k)
    {
      const size_t linidx = (j*gsize[0] +i)*nz_hat + k;
      const long kx = (i <= nKx/2) ? i : -(nKx-i);
      const long l = local_1_start + j; //memory index plus shift due to decomp
      const long ky = (l <= nKy/2) ? l : -(nKy-l);
      const long kz = (k <= nKz/2) ? k : -(nKz-k);

      const Real rkx = kx*wFacX, rky = ky*wFacY, rkz = kz*wFacZ;
      const Real solutionFactor =  - norm_factor / (rkx*rkx+rky*rky+rkz*rkz);
      in_out[linidx][0] *= solutionFactor;
      in_out[linidx][1] *= solutionFactor;
    }
    //this is sparta!
    if (local_1_start == 0) in_out[0][0] = in_out[0][1] = 0;
  #endif

}

PoissonSolverPeriodic::PoissonSolverPeriodic(SimulationData & s) : PoissonSolver(s)
{
  int supported_threads;
  MPI_Query_thread(&supported_threads);
  if (supported_threads<MPI_THREAD_FUNNELED) {
    fprintf(stderr, "PoissonSolverPeriodic ERROR: MPI implementation does not support threads.\n");
    fflush(0); exit(1);
  }

  const int retval = _FFTW_(init_threads)();
  if(retval==0) {
    fprintf(stderr, "PoissonSolverPeriodic: ERROR: Call to fftw_init_threads() returned zero.\n");
    fflush(0); exit(1);
  }
  const int desired_threads = omp_get_max_threads();
  _FFTW_(plan_with_nthreads)(desired_threads);
  _FFTW_(mpi_init)();

  alloc_local = _FFTW_(mpi_local_size_3d_transposed) (
    gsize[0], gsize[1], gsize[2]/2+1, m_comm,
    &local_n0, &local_0_start, &local_n1, &local_1_start);

  data = _FFTW_(alloc_real)(2*alloc_local);
  data_size = (size_t) myN[0] * (size_t) myN[1] * (size_t) 2*nz_hat;
  stridez = 1; // fast
  stridey = 2*nz_hat;
  stridex = myN[1] * 2*nz_hat; // slow

  fwd = (void*) _FFTW_(mpi_plan_dft_r2c_3d)(gsize[0], gsize[1], gsize[2],
    data, (fft_c *)data, m_comm, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE);
  bwd = (void*) _FFTW_(mpi_plan_dft_c2r_3d)(gsize[0], gsize[1], gsize[2],
    (fft_c *)data, data, m_comm, FFTW_MPI_TRANSPOSED_IN  | FFTW_MEASURE);

  //std::cout <<    bs[0] << " " <<    bs[1] << " " <<    bs[2] << " ";
  //std::cout <<   myN[0] << " " <<   myN[1] << " " <<   myN[2] << " ";
  //std::cout << gsize[0] << " " << gsize[1] << " " << gsize[2] << " ";
  //std::cout << mybpd[0] << " " << mybpd[1] << " " << mybpd[2] << std::endl;
}

void PoissonSolverPeriodic::solve()
{
  sim.startProfiler("FFTW cub2rhs");
  _cub2fftw();
  sim.stopProfiler();

  sim.startProfiler("FFTW r2c");
  _FFTW_(execute)((fft_plan) fwd);
  sim.stopProfiler();

  sim.startProfiler("FFTW solve");
  _solve();
  sim.stopProfiler();

  sim.startProfiler("FFTW c2r");
  _FFTW_(execute)((fft_plan) bwd);
  sim.stopProfiler();
}

PoissonSolverPeriodic::~PoissonSolverPeriodic()
{
  _FFTW_(destroy_plan)((fft_plan) fwd);
  _FFTW_(destroy_plan)((fft_plan) bwd);
  _FFTW_(free)(data);
  _FFTW_(mpi_cleanup)();
}

CubismUP_3D_NAMESPACE_END
#undef MPIREAL

#if 0
Real * dump = _FFTW_(alloc_real)(2*alloc_local);
fft_c *const in_out = (fft_c *) data;
fft_c *const out_in = (fft_c *) dump;
#pragma omp parallel for
for(long j = 0; j<static_cast<long>(local_n1); ++j)
for(long i = 0; i<static_cast<long>(gsize[0]); ++i)
for(long k = 0; k<static_cast<long>(nz_hat);   ++k) {
  const size_t linidx = (i*local_n1 +j)*nz_hat + k;
  const size_t linidy = (j*gsize[0] +i)*nz_hat + k;
  out_in[linidx][0] = in_out[linidy][0];
  out_in[linidx][1] = in_out[linidy][1];
}
std::swap(dump, data);
_fftw2cub();
std::swap(dump, data);
_FFTW_(free)(dump);
#endif
