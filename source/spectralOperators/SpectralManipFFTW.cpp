//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
// Created by Guido Novati (novatig@ethz.ch) and
// Hugues de Laroussilhe (huguesdelaroussilhe@gmail.com).
//

#include "SpectralManipFFTW.h"
#include "../poisson/PoissonSolver_common.h"

#ifndef CUP_SINGLE_PRECISION
#define MPIREAL MPI_DOUBLE
#else
#define MPIREAL MPI_FLOAT
#endif /* CUP_SINGLE_PRECISION */

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

void SpectralManipFFTW::_compute_forcing()
{
  fft_c *const cplxData_u  = (fft_c *) data_u;
  fft_c *const cplxData_v  = (fft_c *) data_v;
  fft_c *const cplxData_w  = (fft_c *) data_w;
  const long nKx = static_cast<long>(gsize[0]);
  const long nKy = static_cast<long>(gsize[1]);
  const long nKz = static_cast<long>(gsize[2]);
  const Real waveFactorX = 2.0 * M_PI / sim.extent[0];
  const Real waveFactorY = 2.0 * M_PI / sim.extent[1];
  const Real waveFactorZ = 2.0 * M_PI / sim.extent[2];
  const Real h = sim.uniformH();

  const long loc_n1 = local_n1, shifty = local_1_start;
  const long sizeX = gsize[0], sizeZ_hat = nz_hat;
  const size_t nBins = stats.nBin;
  const long nyquist = stats.nyquist;
  const Real nyquist_scaling = (nyquist-1) / (Real) nyquist;
  assert(nyquist > 0 && nyquist_scaling > 0);

  Real tke = 0, eps = 0, lIntegral = 0, tkeFiltered = 0;
  Real * E_msr = stats.E_msr;
  memset(E_msr, 0, nBins * sizeof(Real)); //Only measure spectrum up to Nyquist
  #ifdef ENERGY_FLUX_SPECTRUM
    Real * Eflux = stats.Eflux;
    memset(Eflux, 0, nBins * sizeof(Real));
    fft_c *const cplxData_j  = (fft_c *) data_j;
  #endif

  #ifdef ENERGY_FLUX_SPECTRUM
    #pragma omp parallel for reduction(+ : E_msr[:nBins], Eflux[:nBins], tke, eps, lIntegral, tkeFiltered) schedule(static)
  #else
    #pragma omp parallel for reduction(+ : E_msr[:nBins], tke, eps, lIntegral, tkeFiltered) schedule(static)
  #endif
  for(long j = 0; j<loc_n1; ++j)
  for(long i = 0; i<sizeX;  ++i)
  for(long k = 0; k<sizeZ_hat; ++k)
  {
    const long linidx = (j*sizeX +i)*sizeZ_hat + k;
    const long ii = (i <= nKx/2) ? i : i-nKx;
    const long l = shifty + j; //memory index plus shift due to decomp
    const long jj = (l <= nKy/2) ? l : l-nKy;
    const long kk = (k <= nKz/2) ? k : k-nKz;

    const Real kx = ii*waveFactorX, ky = jj*waveFactorY, kz = kk*waveFactorZ;
    const Real mult = (k==0) or (k==nKz/2) ? 1 : 2;
    const Real k2 = kx*kx + ky*ky + kz*kz;
    const Real dXfac = 2*std::sin(h * kx);
    const Real dYfac = 2*std::sin(h * ky);
    const Real dZfac = 2*std::sin(h * kz);
    const Real UR = cplxData_u[linidx][0], UI = cplxData_u[linidx][1];
    const Real VR = cplxData_v[linidx][0], VI = cplxData_v[linidx][1];
    const Real WR = cplxData_w[linidx][0], WI = cplxData_w[linidx][1];
    const Real dUdYR = - UI * dYfac, dUdYI = UR * dYfac;
    const Real dUdZR = - UI * dZfac, dUdZI = UR * dZfac;
    const Real dVdXR = - VI * dXfac, dVdXI = VR * dXfac;
    const Real dVdZR = - VI * dZfac, dVdZI = VR * dZfac;
    const Real dWdXR = - WI * dXfac, dWdXI = WR * dXfac;
    const Real dWdYR = - WI * dYfac, dWdYI = WR * dYfac;
    const Real OMGXR = dWdYR - dVdZR, OMGXI = dWdYI - dVdZI;
    const Real OMGYR = dUdZR - dWdXR, OMGYI = dUdZI - dWdXI;
    const Real OMGZR = dVdXR - dUdYR, OMGZI = dVdXI - dUdYI;
    const Real E = mult/2 * (UR*UR + UI*UI + VR*VR + VI*VI + WR*WR + WI*WI);

    tke += (k2 > 0) ? E : 0; // Total kinetic energy
    //eps += k2 * mult/2 * E; // Dissipation rate
    eps += mult/2 * ( OMGXR*OMGXR + OMGXI*OMGXI
                    + OMGYR*OMGYR + OMGYI*OMGYI
                    + OMGZR*OMGZR + OMGZI*OMGZI);    // Dissipation rate
    lIntegral += (k2 > 0) ? E / std::sqrt(k2) : 0; // Large eddy length scale

    const long kind = ii*ii + jj*jj + kk*kk;
    if (kind > 0 && kind < nyquist * nyquist) {
      const size_t binID = std::floor(std::sqrt((Real) kind) * nyquist_scaling);
      assert(binID < nBins);
      E_msr[binID] += E;
      #ifdef ENERGY_FLUX_SPECTRUM
        const Real JR = cplxData_j[linidx][0], JI = cplxData_j[linidx][1];
        #if ENERGY_FLUX_SPECTRUM == 1
          Eflux[binID] += mult*std::sqrt(std::max(JR*JR + JI+JI, (Real)0));
        #else
          const Real S11sqR = pow2(- UI * dXfac),  S11sqI = pow2(UR * dXfac);
          const Real S22sqR = pow2(- VI * dYfac),  S22sqI = pow2(VR * dYfac);
          const Real S33sqR = pow2(- WI * dZfac),  S33sqI = pow2(WR * dZfac);
          const Real S12sqR = pow2(dUdYR+dVdXR), S12sqI = pow2(dUdYI+dVdXI);
          const Real S23sqR = pow2(dVdZR+dWdYR), S23sqI = pow2(dVdZI+dWdYI);
          const Real S31sqR = pow2(dWdXR+dUdZR), S31sqI = pow2(dWdXI+dUdZI);
          const Real SR2 = 2*S11sqR +2*S22sqR +2*S33sqR +S12sqR +S23sqR +S31sqR;
          const Real SI2 = 2*S11sqI +2*S22sqI +2*S33sqI +S12sqI +S23sqI +S31sqI;
          Eflux[binID] += mult*h*h*(JR*std::pow(SR2,1.5) +JI*std::pow(SI2,1.5));
        #endif
      #endif
    }

    if (k2 > 0 && k2 < 3.5) {
      // force first modes, 0.5 added to ensure inclusion of 1^2 + 1^2 + 1^2
      tkeFiltered += E;
    } else {
      cplxData_u[linidx][0] = 0; cplxData_u[linidx][1] = 0;
      cplxData_v[linidx][0] = 0; cplxData_v[linidx][1] = 0;
      cplxData_w[linidx][0] = 0; cplxData_w[linidx][1] = 0;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, E_msr, nBins, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, &tke, 1, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, &eps, 1, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, &lIntegral, 1, MPIREAL, MPI_SUM, m_comm);
  MPI_Allreduce(MPI_IN_PLACE, &tkeFiltered, 1, MPIREAL, MPI_SUM, m_comm);
  #ifdef ENERGY_FLUX_SPECTRUM
  MPI_Allreduce(MPI_IN_PLACE, Eflux, nBins, MPIREAL, MPI_SUM, m_comm);
  #endif

  const Real normalization = 1.0 / pow2(normalizeFFT);
  for (size_t binID = 0; binID < nBins; binID++) {
    E_msr[binID] *= normalization;
    #ifdef ENERGY_FLUX_SPECTRUM
    Eflux[binID] *= normalization;
    #endif
  }

  stats.tke = tke * normalization;
  stats.l_integral = lIntegral / tke;
  stats.tke_filtered = tkeFiltered * normalization;
  stats.dissip_visc = 2 * eps * sim.nu * normalization / pow2(2*h);
  //const Real tau_integral =
  //  lIntegral * M_PI / (2*pow3(stats.uprime)) / pow2(normalizeFFT);

  //stats.dissip_visc = eps * 2 * sim.nu / pow2(normalizeFFT);
}

void SpectralManipFFTW::_compute_IC(const std::vector<Real> &K,
                                        const std::vector<Real> &E)
{
  std::random_device seed;
  const int nthreads = omp_get_max_threads();
  std::vector<std::mt19937> gens(nthreads);
  gens[0] = std::mt19937(seed());
  for(int i=1; i<nthreads; ++i) gens[i] = std::mt19937(gens[0]());

  const EnergySpectrum target(K, E);
  fft_c *const cplxData_u = (fft_c *) data_u;
  fft_c *const cplxData_v = (fft_c *) data_v;
  fft_c *const cplxData_w = (fft_c *) data_w;
  const long nKx = static_cast<long>(gsize[0]);
  const long nKy = static_cast<long>(gsize[1]);
  const long nKz = static_cast<long>(gsize[2]);
  const Real waveFactorX = 2.0 * M_PI / sim.extent[0];
  const Real waveFactorY = 2.0 * M_PI / sim.extent[1];
  const Real waveFactorZ = 2.0 * M_PI / sim.extent[2];
  const long sizeX = gsize[0], sizeZ_hat = nz_hat;
  const long loc_n1 = local_n1, shifty = local_1_start;

  const size_t nBins = stats.nBin;
  const long nyquist = stats.nyquist;
  const Real nyquist_scaling = (nyquist-1) / (Real) nyquist;
  assert(nyquist > 0 && nyquist_scaling > 0);

  Real * E_msr = new Real[nBins];
  memset(E_msr, 0, nBins * sizeof(Real));

  #pragma omp parallel reduction(+ : E_msr[:nBins])
  {
    std::mt19937 & gen = gens[omp_get_thread_num()];
    std::uniform_real_distribution<Real> randUniform(0,1);

    #pragma omp for schedule(static)
    for(long j = 0; j<loc_n1;  ++j)
    for(long i = 0; i<sizeX;    ++i)
    for(long k = 0; k<sizeZ_hat; ++k)
    {
      const long linidx = (j*sizeX +i) * sizeZ_hat + k;
      const long ii = (i <= nKx/2) ? i : i-nKx;
      const long l = shifty + j; //memory index plus shift due to decomp
      const long jj = (l <= nKy/2) ? l : l-nKy;
      const long kk = (k <= nKz/2) ? k : k-nKz;
      const Real mult = (k==0) or (k==nKz/2) ? 1 : 2;

      const Real kx = ii*waveFactorX, ky = jj*waveFactorY, kz = kk*waveFactorZ;
      const Real k2 = kx*kx + ky*ky + kz*kz;
      const Real k_xy = std::sqrt(kx*kx + ky*ky);
      const Real k_norm = std::sqrt(k2);

      const Real E_k = target.interpE(k_norm);
      //const Real amp = (k2<=0)? 0 // hugues:
      //                : std::sqrt(E_k/(2*M_PI*std::pow(k_norm/waveFactorX,2)));
      const Real amp = (k2<=0)? 0 : std::sqrt(E_k/(2*M_PI*k_norm*k_norm));
      const Real theta1 = randUniform(gen)*2*M_PI;
      const Real theta2 = randUniform(gen)*2*M_PI;
      const Real phi    = randUniform(gen)*2*M_PI;

      const Real alpha_r = amp * std::cos(theta1) * std::cos(phi);
      const Real alpha_i = amp * std::sin(theta1) * std::cos(phi);
      const Real beta_r = amp * std::cos(theta2) * std::sin(phi);
      const Real beta_i = amp * std::sin(theta2) * std::sin(phi);

      const Real fac = k_norm*k_xy, invFac = fac<=0? 0 : 1/fac;

      cplxData_u[linidx][0] = k_norm<=0? 0
                    : invFac * (alpha_r * k_norm*ky + beta_r * kx*kz );
      cplxData_u[linidx][1] = k_norm<=0? 0
                    : invFac * (alpha_i * k_norm*ky + beta_i * kx*kz );

      cplxData_v[linidx][0] = k_norm<=0? 0
                    : invFac * (beta_r * ky*kz - alpha_r * k_norm*kx );
      cplxData_v[linidx][1] = k_norm<=0? 0
                    : invFac * (beta_i * ky*kz - alpha_i * k_norm*kx );

      cplxData_w[linidx][0] = k_norm<=0? 0 : -beta_r * k_xy / k_norm;
      cplxData_w[linidx][1] = k_norm<=0? 0 : -beta_i * k_xy / k_norm;

      const long kind = ii*ii + jj*jj + kk*kk;
      if (kind > 0 && kind < nyquist * nyquist) {
        const Real UR = cplxData_u[linidx][0], UI = cplxData_u[linidx][1];
        const Real VR = cplxData_v[linidx][0], VI = cplxData_v[linidx][1];
        const Real WR = cplxData_w[linidx][0], WI = cplxData_w[linidx][1];
        const size_t binID = std::floor(std::sqrt((Real) kind)*nyquist_scaling);
        assert(binID < nBins);
        E_msr[binID] += mult/2 *(UR*UR + UI*UI + VR*VR + VI*VI + WR*WR + WI*WI);
      }
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, E_msr, nBins, MPIREAL, MPI_SUM, m_comm);

  // perform back and forth transform twice to ensure we match spectrum
  for (int iter=0; iter < 2; iter++)
  {
    #pragma omp parallel for schedule(static)
    for(long j = 0; j<loc_n1;  ++j)
    for(long i = 0; i<sizeX;    ++i)
    for(long k = 0; k<sizeZ_hat; ++k) {
      const long linidx = (j*sizeX +i) * sizeZ_hat + k;
      const long ii = (i <= nKx/2) ? i : i-nKx;
      const long l = shifty + j; //memory index plus shift due to decomp
      const long jj = (l <= nKy/2) ? l : l-nKy;
      const long kk = (k <= nKz/2) ? k : k-nKz;

      const long kind = ii*ii + jj*jj + kk*kk;
      if (kind > 0 && kind < nyquist * nyquist) {
        const size_t binID = std::floor(std::sqrt((Real) kind) * nyquist_scaling);
        assert(binID < nBins);
        const Real fac = E_msr[binID]>0? std::sqrt(E[binID] / E_msr[binID]) : 0;
        cplxData_u[linidx][0] *= fac; cplxData_u[linidx][1] *= fac;
        cplxData_v[linidx][0] *= fac; cplxData_v[linidx][1] *= fac;
        cplxData_w[linidx][0] *= fac; cplxData_w[linidx][1] *= fac;
      }
    }

    runBwd();
    runFwd();
    //std::cout << iter << " E pre:";
    //for (int i = 0; i < nBins; i++) std::cout << E_msr[i] << " ";
    //std::cout << std::endl;

    memset(E_msr, 0, nBins * sizeof(Real));
    #pragma omp parallel for schedule(static) reduction(+ : E_msr[:nBins])
    for(long j = 0; j<loc_n1;  ++j)
    for(long i = 0; i<sizeX;    ++i)
    for(long k = 0; k<sizeZ_hat; ++k) {
      const long linidx = (j*sizeX +i) * sizeZ_hat + k;
      const long ii = (i <= nKx/2) ? i : -(nKx-i);
      const long l = shifty + j; //memory index plus shift due to decomp
      const long jj = (l <= nKy/2) ? l : -(nKy-l);
      const long kk = (k <= nKz/2) ? k : -(nKz-k);
      const Real mult = (k==0) or (k==nKz/2) ? 1 : 2;
      cplxData_u[linidx][0] /= normalizeFFT; // after each runFwd we need
      cplxData_v[linidx][0] /= normalizeFFT; // to renormalize
      cplxData_w[linidx][0] /= normalizeFFT;
      cplxData_u[linidx][1] /= normalizeFFT;
      cplxData_v[linidx][1] /= normalizeFFT;
      cplxData_w[linidx][1] /= normalizeFFT;

      const Real UR = cplxData_u[linidx][0], UI = cplxData_u[linidx][1];
      const Real VR = cplxData_v[linidx][0], VI = cplxData_v[linidx][1];
      const Real WR = cplxData_w[linidx][0], WI = cplxData_w[linidx][1];

      const long kind = ii*ii + jj*jj + kk*kk;
      if (kind > 0 && kind < nyquist * nyquist) {
        const size_t binID = std::floor(std::sqrt((Real) kind)*nyquist_scaling);
        assert(binID < nBins);
        E_msr[binID] += mult/2 * (UR*UR + UI*UI + VR*VR + VI*VI + WR*WR + WI*WI);
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, E_msr, nBins, MPIREAL, MPI_SUM, m_comm);
    //std::cout << "E post:";
    //for (int i = 0; i < nBins; i++) std::cout << E_msr[i] << " ";
    //std::cout << std::endl;
  }

  delete [] E_msr;
}

SpectralManipFFTW::SpectralManipFFTW(SimulationData&s): SpectralManip(s)
{
  const int retval = _FFTW_(init_threads)();
  if(retval==0) {
    fprintf(stderr, "SpectralManip: ERROR: Call to fftw_init_threads() returned zero.\n");
    fflush(0); exit(1);
  }
  _FFTW_(mpi_init)();
  const int desired_threads = omp_get_max_threads();
  _FFTW_(plan_with_nthreads)(desired_threads);

  alloc_local = _FFTW_(mpi_local_size_3d_transposed) (
    gsize[0], gsize[1], gsize[2]/2+1, m_comm,
    &local_n0, &local_0_start, &local_n1, &local_1_start);

  data_size = (size_t) myN[0] * (size_t) myN[1] * (size_t) 2*nz_hat;
  stridez = 1; // fast
  stridey = 2*(nz_hat);
  stridex = myN[1] * 2*(nz_hat); // slow

  data_u = _FFTW_(alloc_real)(2*alloc_local);
  data_v = _FFTW_(alloc_real)(2*alloc_local);
  data_w = _FFTW_(alloc_real)(2*alloc_local);

  #ifdef ENERGY_FLUX_SPECTRUM
    data_j = _FFTW_(alloc_real)(2*alloc_local);
  #endif
}

void SpectralManipFFTW::prepareFwd()
{
  if (bAllocFwd) return;

  fwd_u = (void*) _FFTW_(mpi_plan_dft_r2c_3d)(gsize[0], gsize[1], gsize[2],
    data_u, (fft_c*)data_u, m_comm, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE);
  fwd_v = (void*) _FFTW_(mpi_plan_dft_r2c_3d)(gsize[0], gsize[1], gsize[2],
    data_v, (fft_c*)data_v, m_comm, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE);
  fwd_w = (void*) _FFTW_(mpi_plan_dft_r2c_3d)(gsize[0], gsize[1], gsize[2],
    data_w, (fft_c*)data_w, m_comm, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE);

  #ifdef ENERGY_FLUX_SPECTRUM
  fwd_j = (void*) _FFTW_(mpi_plan_dft_r2c_3d)(gsize[0], gsize[1], gsize[2],
    data_j, (fft_c*)data_j, m_comm, FFTW_MPI_TRANSPOSED_OUT | FFTW_MEASURE);
  #endif

  bAllocFwd = true;
}

void SpectralManipFFTW::prepareBwd()
{
  if (bAllocBwd) return;

  bwd_u = (void*) _FFTW_(mpi_plan_dft_c2r_3d)(gsize[0], gsize[1], gsize[2],
    (fft_c*)data_u, data_u, m_comm, FFTW_MPI_TRANSPOSED_IN  | FFTW_MEASURE);
  bwd_v = (void*) _FFTW_(mpi_plan_dft_c2r_3d)(gsize[0], gsize[1], gsize[2],
    (fft_c*)data_v, data_v, m_comm, FFTW_MPI_TRANSPOSED_IN  | FFTW_MEASURE);
  bwd_w = (void*) _FFTW_(mpi_plan_dft_c2r_3d)(gsize[0], gsize[1], gsize[2],
    (fft_c*)data_w, data_w, m_comm, FFTW_MPI_TRANSPOSED_IN  | FFTW_MEASURE);

  bAllocBwd = true;
}

void SpectralManipFFTW::runFwd() const
{
  assert(bAllocFwd);
  // we can use one plan for multiple data:
  //_FFTW_(execute_dft_r2c)( (fft_plan) fwd_u, data_u, (fft_c*)data_u );
  //_FFTW_(execute_dft_r2c)( (fft_plan) fwd_u, data_v, (fft_c*)data_v );
  //_FFTW_(execute_dft_r2c)( (fft_plan) fwd_u, data_w, (fft_c*)data_w );
  _FFTW_(execute)((fft_plan) fwd_u);
  _FFTW_(execute)((fft_plan) fwd_v);
  _FFTW_(execute)((fft_plan) fwd_w);

  #ifdef ENERGY_FLUX_SPECTRUM
    _FFTW_(execute)((fft_plan) fwd_j);
  #endif
}

void SpectralManipFFTW::runBwd() const
{
  assert(bAllocBwd);
  // we can use one plan for multiple data:
  //_FFTW_(execute_dft_c2r)( (fft_plan) bwd_u, (fft_c*)data_u, data_u );
  //_FFTW_(execute_dft_c2r)( (fft_plan) bwd_u, (fft_c*)data_v, data_v );
  //_FFTW_(execute_dft_c2r)( (fft_plan) bwd_u, (fft_c*)data_w, data_w );
  _FFTW_(execute)((fft_plan) bwd_u);
  _FFTW_(execute)((fft_plan) bwd_v);
  _FFTW_(execute)((fft_plan) bwd_w);
}

SpectralManipFFTW::~SpectralManipFFTW()
{
  _FFTW_(free)(data_u);
  _FFTW_(free)(data_v);
  _FFTW_(free)(data_w);
  #ifdef ENERGY_FLUX_SPECTRUM
    _FFTW_(free)(data_j);
  #endif
  if (bAllocFwd) {
    _FFTW_(destroy_plan)((fft_plan) fwd_u);
    _FFTW_(destroy_plan)((fft_plan) fwd_v);
    _FFTW_(destroy_plan)((fft_plan) fwd_w);
    #ifdef ENERGY_FLUX_SPECTRUM
      _FFTW_(destroy_plan)((fft_plan) fwd_j);
    #endif
  }
  if (bAllocBwd) {
    _FFTW_(destroy_plan)((fft_plan) bwd_u);
    _FFTW_(destroy_plan)((fft_plan) bwd_v);
    _FFTW_(destroy_plan)((fft_plan) bwd_w);
  }

  _FFTW_(mpi_cleanup)();
}

CubismUP_3D_NAMESPACE_END
#undef MPIREAL
