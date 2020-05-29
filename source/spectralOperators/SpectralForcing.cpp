//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Hugues de Laroussilhe.
//

#include "SpectralForcing.h"
#include "SpectralManip.h"
#include "../utils/BufferedLogger.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

SpectralForcing::SpectralForcing(SimulationData & s) : Operator(s)
{
  initSpectralAnalysisSolver(s);
  s.spectralManip->prepareFwd();
  s.spectralManip->prepareBwd();
}

void SpectralForcing::operator()(const double dt)
{
  sim.startProfiler("SpectralForcing");
  SpectralManip & sM = * sim.spectralManip;
  HITstatistics & stats = sM.stats;

  _cub2fftw();

  sM.runFwd();

  sM._compute_forcing();

  sM.runBwd();

  sim.actualInjectionRate = 0;
  //With non spectral IC, the target tke may not be defined here
  if      (sim.turbKinEn_target > 0) // inject energy to match target tke
       sim.actualInjectionRate = (sim.turbKinEn_target - stats.tke)/dt;
  else if (sim.enInjectionRate  > 0) // constant power input:
       sim.actualInjectionRate =  sim.enInjectionRate;

  stats.updateDerivedQuantities(sim.nu, dt, sim.actualInjectionRate);
  const Real fac = dt * sim.actualInjectionRate / (2 * stats.tke_filtered);

  if(fac>0) _fftw2cub(fac / sM.normalizeFFT);
  stats.expectedNextTke = stats.tke + dt * sim.actualInjectionRate;
  // If there's too much energy, let dissipation do its job
  if(sim.verbose)
    printf("totalKinEn:%e largeModesKinEn:%e injectionRate:%e viscousDissip:%e totalDissipRate:%e lIntegral:%e\n",
    stats.tke, stats.tke_filtered, sim.actualInjectionRate,
    stats.dissip_visc, stats.dissip_tot, stats.l_integral);

  if(sim.rank == 0 and not sim.muteAll) {
    std::stringstream &ssF = logger.get_stream("forcingData.dat");
    const std::string tab("\t");
    if(sim.step==0) {
      ssF<<"step \t time \t dt \t totalKinEn \t largeModesKinEn \t "\
           "viscousDissip \t totalDissipRate \t injectionRate \t lIntegral\n";
    }

    ssF << sim.step << tab;
    ssF.setf(std::ios::scientific);
    ssF.precision(std::numeric_limits<float>::digits10 + 1);
    ssF<<sim.time<<tab<<sim.dt<<tab<<stats.tke<<tab<<stats.tke_filtered<<tab
       <<stats.dissip_visc<<tab<<stats.dissip_tot<<tab<<sim.actualInjectionRate
       <<tab<<stats.l_integral<<"\n";
  }

  sim.stopProfiler();

  check("SpectralForcing");
}

void SpectralForcing::_cub2fftw() const
{
  const SpectralManip& sM = * sim.spectralManip;
  const size_t NlocBlocks = sM.local_infos.size();
  Real * const data_u = sM.data_u;
  Real * const data_v = sM.data_v;
  Real * const data_w = sM.data_w;

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<NlocBlocks; ++i) {
    const FluidBlock& b = *(FluidBlock*) sM.local_infos[i].ptrBlock;
    const size_t offset = sM._offset( sM.local_infos[i] );
    for(size_t iz=0; iz < (size_t) FluidBlock::sizeZ; ++iz)
    for(size_t iy=0; iy < (size_t) FluidBlock::sizeY; ++iy)
    for(size_t ix=0; ix < (size_t) FluidBlock::sizeX; ++ix) {
      const size_t src_index = sM._dest(offset, iz, iy, ix);
      data_u[src_index] = b(ix,iy,iz).u;
      data_v[src_index] = b(ix,iy,iz).v;
      data_w[src_index] = b(ix,iy,iz).w;
    }
  }
}

void SpectralForcing::_fftw2cub(const Real factor) const
{
  const SpectralManip& sM = * sim.spectralManip;
  const size_t NlocBlocks = sM.local_infos.size();
  const Real * const data_u = sM.data_u;
  const Real * const data_v = sM.data_v;
  const Real * const data_w = sM.data_w;

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<NlocBlocks; ++i) {
    FluidBlock& b = *(FluidBlock*) sM.local_infos[i].ptrBlock;
    const size_t offset = sM._offset( sM.local_infos[i] );
    for(size_t iz=0; iz< (size_t) FluidBlock::sizeZ; ++iz)
    for(size_t iy=0; iy< (size_t) FluidBlock::sizeY; ++iy)
    for(size_t ix=0; ix< (size_t) FluidBlock::sizeX; ++ix) {
      const size_t src_index = sM._dest(offset, iz, iy, ix);
      b(ix,iy,iz).u += factor * data_u[src_index];
      b(ix,iy,iz).v += factor * data_v[src_index];
      b(ix,iy,iz).w += factor * data_w[src_index];
    }
  }
}

CubismUP_3D_NAMESPACE_END
