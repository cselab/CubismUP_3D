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
#ifndef CubismUP_3D_PoissonSolverUnbounded_h
#define CubismUP_3D_PoissonSolverUnbounded_h

#include "PoissonSolver.h"

CubismUP_3D_NAMESPACE_BEGIN

class PoissonSolverUnbounded : public PoissonSolver
{
  // domain dimensions
  const size_t m_N0 = gsize[0];  // Nx-points of original domain
  const size_t m_N1 = gsize[1];  // Ny-points of original domain
  const size_t m_N2 = gsize[2];  // Nz-points of original domain
  const size_t m_NN0 = 2*m_N0; // Nx-points of padded domain
  const size_t m_NN1 = 2*m_N1; // Ny-points of padded domain
  const size_t m_NN2 = 2*m_N2; // Nz-points of padded domain
  const size_t m_NN0t = 2*m_N0-1; //Nx of padded domain (w/o periodic copies, actual transform size)
  const size_t m_NN1t = 2*m_N1-1; //Ny of padded domain (w/o periodic copies, actual transform size)
  const size_t m_NN2t = 2*m_N2-1; //Nz of padded domain (w/o periodic copies, actual transform size)

  const size_t m_local_N0 = m_N0 / m_size;
  //const size_t m_start_N0 = m_local_N0 * m_rank;
  const size_t m_local_NN0 = m_NN0/m_size, m_start_NN0 = m_local_NN0*m_rank;
  const size_t m_local_NN1 = m_NN1/m_size;// m_start_NN1 = m_local_NN1*m_rank;
  const double h = sim.uniformH();
  // data buffers for input and transform.  Split into 2 buffers to exploit
  // smaller transpose matrix and fewer FFT's due to zero-padded domain.
  // This is at the cost of higher memory requirements.
  const size_t m_Nzhat = m_NN2t/2 + 1; // for symmetry in r2c transform
  const size_t m_tp_size   = m_local_N0  * m_NN1  * m_Nzhat;
  const size_t m_full_size = m_NN0t * m_local_NN1 * m_Nzhat;
  // FFT normalization factor
  const Real m_norm_factor = 1.0 / (m_NN0t*h * m_NN1t*h * m_NN2t*h);
  Real* m_buf_full; // full block of m_NN0t x m_local_NN1 x 2m_Nzhat for 1D FFTs
  Real* m_kernel;   // FFT of Green's function (real part, m_NN0t x m_local_NN1 x m_Nzhat)

  // FFTW plans
  void * m_fwd_1D;
  void * m_bwd_1D;
  void * m_fwd_2D;
  void * m_bwd_2D;
  void * m_fwd_tp; // use FFTW's transpose facility
  void * m_bwd_tp; // use FFTW's transpose facility

 public:
  PoissonSolverUnbounded(SimulationData&s);

  PoissonSolverUnbounded(const PoissonSolverUnbounded& c) = delete;
  ~PoissonSolverUnbounded() override;

  void solve() override;

 private:

  void _initialize_green();

  void reset() const override;

  void _copy_fwd_local();

  void _copy_bwd_local();
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_PoissonSolverUnbounded_h
