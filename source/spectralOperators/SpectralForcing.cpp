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

#if defined(ENERGY_FLUX_SPECTRUM) && ENERGY_FLUX_SPECTRUM == 1
struct KernelComputeEnergyFlux
{
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W}};

  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o) const
  {
    const Real h = info.h_gridpoint, fac = 1.0/(4*h*h);
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);
      const Real dudx = LE.u-LW.u, dvdx = LE.v-LW.v, dwdx = LE.w-LW.w;
      const Real dudy = LN.u-LS.u, dvdy = LN.v-LS.v, dwdy = LN.w-LS.w;
      const Real dudz = LB.u-LF.u, dvdz = LB.v-LF.v, dwdz = LB.w-LF.w;
      const Real SijSij2 = (2*dudx*dudx + (dudy+dvdx)*(dudy+dvdx)
                          + 2*dvdy*dvdy + (dudz+dwdx)*(dudz+dwdx)
                          + 2*dwdz*dwdz + (dwdy+dvdz)*(dwdy+dvdz)) * fac;
      o(ix,iy,iz).tmpU = o(ix,iy,iz).chi * h*h * std::pow(SijSij2, 1.5);
    }
  }
};
#endif

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

  #if defined(ENERGY_FLUX_SPECTRUM) && ENERGY_FLUX_SPECTRUM == 1
    const KernelComputeEnergyFlux K;
    compute(K);
  #endif

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
  #ifdef ENERGY_FLUX_SPECTRUM
    Real * const data_j = sM.data_j;
  #endif

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
      #ifdef ENERGY_FLUX_SPECTRUM
        #if ENERGY_FLUX_SPECTRUM == 1
          data_j[src_index] = b(ix,iy,iz).tmpU;
        #else
          data_j[src_index] = b(ix,iy,iz).chi;
        #endif
      #endif
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
