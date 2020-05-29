//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Written by Fabian Wermelinger.
//
//  This algorithm uses the cyclic convolution method described in Eastwood and
//  Brownrigg (1979) for unbounded domains.
//  WARNING: This implementation only works with a 1D domain decomposition
//  along the x-coordinate.

#include "PoissonSolverUnbounded.h"
#include "PoissonSolver_common.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

PoissonSolverUnbounded::PoissonSolverUnbounded(SimulationData&s) : PoissonSolver(s)
{
  if (m_N0 % m_size != 0 || m_NN1 % m_size != 0) {
    fprintf(stderr, "PoissonSolverUnbounded: ERROR: Number of cells N0 and 2*N1 must be evenly divisible by the number of processes.\n");
    fflush(0); exit(1);
  }

  int supported_threads;
  MPI_Query_thread(&supported_threads);
  if (supported_threads < MPI_THREAD_FUNNELED) {
    fprintf(stderr, "PoissonSolverUnbounded: ERROR: MPI implementation does not support threads.\n");
    fflush(0); exit(1);
  }

  const int retval = _FFTW_(init_threads)();
  if (retval == 0) {
    fprintf(stderr, "PoissonSolverUnbounded: ERROR: Call to fftw_init_threads() returned zero.\n");
    fflush(0); exit(1);
  }
  const int desired_threads = omp_get_max_threads();
  _FFTW_(plan_with_nthreads)(desired_threads);
  _FFTW_(mpi_init)();

  _initialize_green();

  // FFTW plans
  // input, output, transpose and 2D FFTs (m_local_N0 x m_NN1 x 2m_Nzhat):
  data   = _FFTW_(alloc_real)( 2*m_tp_size );
  data_size = (size_t) m_local_N0 * (size_t) m_NN1 * (size_t) 2*m_Nzhat;
  stridez = 1; // fast
  stridey = 2*m_Nzhat;
  stridex = m_NN1 * 2*m_Nzhat; // slow

  m_buf_full = _FFTW_(alloc_real)( 2*m_full_size );

  // 1D plan
  {
    const int n[1] = {static_cast<int>(m_NN0t)};
    const int howmany = static_cast<int>( m_local_NN1 * m_Nzhat );
    const int stride  = static_cast<int>( m_local_NN1 * m_Nzhat );
    const int* embed = n;
    const int dist = 1;
    m_fwd_1D = (void*) _FFTW_(plan_many_dft)(1, n, howmany,
            (fft_c*)m_buf_full, embed, stride, dist,
            (fft_c*)m_buf_full, embed, stride, dist,
            FFTW_FORWARD, FFTW_MEASURE);
    m_bwd_1D = (void*) _FFTW_(plan_many_dft)(1, n, howmany,
            (fft_c*)m_buf_full, embed, stride, dist,
            (fft_c*)m_buf_full, embed, stride, dist,
            FFTW_BACKWARD, FFTW_MEASURE);
  }

  // 2D plan
  {
    const int n[2] = {static_cast<int>(m_NN1t), static_cast<int>(m_NN2t)};
    const int howmany = static_cast<int>(m_local_N0);
    const int stride = 1;
    const int rembed[2] = {static_cast<int>(m_NN1), static_cast<int>(2*m_Nzhat)}; // unit: sizeof(Real)
    const int cembed[2] = {static_cast<int>(m_NN1), static_cast<int>(m_Nzhat)};   // unit: sizeof(fft_c)
    const int rdist = static_cast<int>( m_NN1 * 2*m_Nzhat ); // unit: sizeof(Real)
    const int cdist = static_cast<int>( m_NN1 * m_Nzhat ); // unit: sizeof(fft_c)
    m_fwd_2D = (void*) _FFTW_(plan_many_dft_r2c)(2, n, howmany,
            data, rembed, stride, rdist,
            (fft_c*)data, cembed, stride, cdist,
            FFTW_MEASURE);
    m_bwd_2D = (void*) _FFTW_(plan_many_dft_c2r)(2, n, howmany,
            (fft_c*)data, cembed, stride, cdist,
            data, rembed, stride, rdist,
            FFTW_MEASURE);
  }

  // transpose plan
  m_fwd_tp = (void*) _FFTW_(mpi_plan_many_transpose)(m_N0, m_NN1, 2*m_Nzhat,
          m_local_N0, m_local_NN1, data, data, m_comm,
          FFTW_MEASURE | FFTW_MPI_TRANSPOSED_OUT);

  m_bwd_tp = (void*) _FFTW_(mpi_plan_many_transpose)(m_NN1, m_N0, 2*m_Nzhat,
          m_local_NN1, m_local_N0, data, data, m_comm,
          FFTW_MEASURE | FFTW_MPI_TRANSPOSED_IN);
}

PoissonSolverUnbounded::~PoissonSolverUnbounded()
{
  _FFTW_(free)(data);
  _FFTW_(free)(m_buf_full);
  _FFTW_(free)(m_kernel);
  _FFTW_(destroy_plan)((fft_plan) m_fwd_1D);
  _FFTW_(destroy_plan)((fft_plan) m_bwd_1D);
  _FFTW_(destroy_plan)((fft_plan) m_fwd_2D);
  _FFTW_(destroy_plan)((fft_plan) m_bwd_2D);
  _FFTW_(destroy_plan)((fft_plan) m_fwd_tp);
  _FFTW_(destroy_plan)((fft_plan) m_bwd_tp);
  _FFTW_(mpi_cleanup)();
}

void PoissonSolverUnbounded::solve()
{
  sim.startProfiler("UFFTW cub2rhs");
  _cub2fftw();
  sim.stopProfiler();

  sim.startProfiler("UFFTW r2c");
  _FFTW_(execute)((fft_plan) m_fwd_2D);
  sim.stopProfiler();
  _FFTW_(execute)((fft_plan) m_fwd_tp);
  _copy_fwd_local();
  _FFTW_(execute)((fft_plan) m_fwd_1D);

  sim.startProfiler("UFFTW solve");
  {
    fft_c* const rho_hat = (fft_c*)m_buf_full;
    const Real* const G_hat = m_kernel;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < m_NN0t; ++i)
    for (size_t j = 0; j < m_local_NN1; ++j)
    for (size_t k = 0; k < m_Nzhat; ++k)
    {
      const size_t idx = k + m_Nzhat*(j + m_local_NN1*i);
      rho_hat[idx][0] *= G_hat[idx]; //normalization is carried on in G_hat
      rho_hat[idx][1] *= G_hat[idx]; //normalization is carried on in G_hat
    }
  }
  sim.stopProfiler();

  sim.startProfiler("UFFTW c2r");
  _FFTW_(execute)((fft_plan) m_bwd_1D);
  sim.stopProfiler();
  _copy_bwd_local();
  _FFTW_(execute)((fft_plan) m_bwd_tp);
  _FFTW_(execute)((fft_plan) m_bwd_2D);
}


void PoissonSolverUnbounded::_initialize_green()
{
  fft_plan green1D;
  fft_plan green2D;
  fft_plan greenTP;

  const size_t tf_size = m_local_NN0 * m_NN1 * m_Nzhat;
  Real* tf_buf = _FFTW_(alloc_real)( 2*tf_size );

  // 1D plan
  {
  const int n[1] = {static_cast<int>(m_NN0t)};
  const int howmany = static_cast<int>( m_local_NN1 * m_Nzhat );
  const int stride  = static_cast<int>( m_local_NN1 * m_Nzhat );
  const int* embed = n;
  const int dist = 1;
  green1D = _FFTW_(plan_many_dft)(1, n, howmany,
          (fft_c*)tf_buf, embed, stride, dist,
          (fft_c*)tf_buf, embed, stride, dist,
          FFTW_FORWARD, FFTW_MEASURE);
  }

  // 2D plan
  {
  const int n[2] = {static_cast<int>(m_NN1t), static_cast<int>(m_NN2t)};
  const int howmany = static_cast<int>(m_local_NN0);
  const int stride = 1;
  const int rembed[2] = {static_cast<int>(m_NN1), static_cast<int>(2*m_Nzhat)}; // unit: sizeof(Real)
  const int cembed[2] = {static_cast<int>(m_NN1), static_cast<int>(m_Nzhat)};   // unit: sizeof(fft_c)
  const int rdist = static_cast<int>( m_NN1 * 2*m_Nzhat );                      // unit: sizeof(Real)
  const int cdist = static_cast<int>( m_NN1 * m_Nzhat );                        // unit: sizeof(fft_c)
  green2D = _FFTW_(plan_many_dft_r2c)(2, n, howmany,
          tf_buf, rembed, stride, rdist,
          (fft_c*)tf_buf, cembed, stride, cdist,
          FFTW_MEASURE);
  }

  greenTP = _FFTW_(mpi_plan_many_transpose)(m_NN0, m_NN1, 2*m_Nzhat,
          m_local_NN0, m_local_NN1,
          tf_buf, tf_buf,
          m_comm, FFTW_MEASURE|FFTW_MPI_TRANSPOSED_OUT);

  // This factor is due to the discretization of the convolution
  // integtal.  It is composed of (h*h*h) * (-1/[4*pi*h]), where h is the
  // uniform grid spacing.  The first factor is the discrete volume
  // element of the convolution integral; the second factor belongs to
  // Green's function on a uniform mesh.
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m_local_NN0; ++i)
  for (size_t j = 0; j < m_NN1; ++j)
  for (size_t k = 0; k < m_NN2; ++k)
  {
      const size_t I = m_start_NN0 + i;
      const Real xi = I>=m_N0? 2*m_N0-1 - I : I;
      const Real yi = j>=m_N1? 2*m_N1-1 - j : j;
      const Real zi = k>=m_N2? 2*m_N2-1 - k : k;
      const double r = std::sqrt(xi*xi + yi*yi + zi*zi);
      const size_t idx = k + 2*m_Nzhat*(j + m_NN1*i);
      if (r > 0) tf_buf[idx] = - h * h / (4*M_PI*r);
      else tf_buf[idx] = - Real(0.1924173658) * h * h;
  }

  _FFTW_(execute)(green2D);
  _FFTW_(execute)(greenTP);
  _FFTW_(execute)(green1D);

  const size_t kern_size = m_NN0t * m_local_NN1 * m_Nzhat;
  m_kernel = _FFTW_(alloc_real)(kern_size); // FFT for this kernel is real
  std::memset(m_kernel, 0, kern_size*sizeof(Real));

  const fft_c *const G_hat = (fft_c *) tf_buf;
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m_NN0t; ++i)
  for (size_t j = 0; j < m_local_NN1; ++j)
  for (size_t k = 0; k < m_Nzhat; ++k)
  {
    const size_t linidx = k + m_Nzhat*(j + m_local_NN1*i);
    m_kernel[linidx] = G_hat[linidx][0] * m_norm_factor;// need real part only
  }

  _FFTW_(free)(tf_buf);
  _FFTW_(destroy_plan)(green1D);
  _FFTW_(destroy_plan)(green2D);
  _FFTW_(destroy_plan)(greenTP);
}

void PoissonSolverUnbounded::reset() const
{
  std::memset(data, 0, 2*m_tp_size*sizeof(Real));
  std::memset(m_buf_full, 0, 2*m_full_size*sizeof(Real));
}

void PoissonSolverUnbounded::_copy_fwd_local()
{
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m_N0; ++i)
  for (size_t j = 0; j < m_local_NN1; ++j) {
    const Real* const src = data + 2*m_Nzhat*(j + m_local_NN1*i);
    Real* const dst = m_buf_full + 2*m_Nzhat*(j + m_local_NN1*i);
    std::memcpy(dst, src, 2*m_Nzhat*sizeof(Real));
  }
}

void PoissonSolverUnbounded::_copy_bwd_local()
{
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < m_N0; ++i)
  for (size_t j = 0; j < m_local_NN1; ++j) {
    const Real* const src = m_buf_full + 2*m_Nzhat*(j + m_local_NN1*i);
    Real* const dst = data + 2*m_Nzhat*(j + m_local_NN1*i);
    std::memcpy(dst, src, 2*m_Nzhat*sizeof(Real));
  }
}

CubismUP_3D_NAMESPACE_END
#undef MPIREAL
