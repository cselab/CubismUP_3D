//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Hugues de Laroussilhe.
//

#include "SpectralAnalysis.h"
#include "SpectralManip.h"
#include "HITtargetData.h"
#include "../operators/ProcessHelpers.h"
#include <Cubism/HDF5Dumper_MPI.h>

#include <sys/stat.h>
#include <iomanip>
#include <sstream>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

#ifndef CUP_SINGLE_PRECISION
#define MPIREAL MPI_DOUBLE
#else
#define MPIREAL MPI_FLOAT
#endif /* CUP_SINGLE_PRECISION */

SpectralAnalysis::SpectralAnalysis(SimulationData & s)
{
  initSpectralAnalysisSolver(s);
  s.spectralManip->prepareFwd();
  s.spectralManip->prepareBwd();
  sM = s.spectralManip;

  target = new HITtargetData(sM->maxGridN, "");
  target->smartiesFolderStructure = false;
  target->readAll("target");
  if (not target->holdsTargetData) {
    delete target;
    target = nullptr;
  } else s.saveTime = target->tInteg;
}

void SpectralAnalysis::_cub2fftw()
{
  // Let's also compute u_avg here
  const size_t NlocBlocks = sM->local_infos.size();
  Real * const data_u = sM->data_u;
  Real * const data_v = sM->data_v;
  Real * const data_w = sM->data_w;
  //Real * const data_cs2 = sM->data_cs2;
  assert(sM not_eq nullptr);
  const SpectralManip & helper = * sM;
  //Real unorm = 0;
  //u_avg[0] = 0; u_avg[1] = 0; u_avg[2] = 0; unorm = 0;
  #pragma omp parallel for schedule(static) // reduction(+: u_avg[:3], unorm)
  for(size_t i=0; i<NlocBlocks; ++i)
  {
    const BlockType& b = *(BlockType*) helper.local_infos[i].ptrBlock;
    const size_t offset = helper._offset( helper.local_infos[i] );
    for(int iz=0; iz<BlockType::sizeZ; ++iz)
    for(int iy=0; iy<BlockType::sizeY; ++iy)
    for(int ix=0; ix<BlockType::sizeX; ++ix)
    {
      const size_t ind = helper._dest(offset, iz, iy, ix);
      data_u[ind] = b(ix,iy,iz).u;
      data_v[ind] = b(ix,iy,iz).v;
      data_w[ind] = b(ix,iy,iz).w;
      //u_avg[0]+= data_u[ind]; u_avg[1]+= data_v[ind]; u_avg[2]+= data_w[ind];
      //unorm += pow2(data_u[ind]) + pow2(data_v[ind]) + pow2(data_w[ind]);
      //data_cs2[src_index] = b(ix,iy,iz).chi;
    }
  }
  //MPI_Allreduce(MPI_IN_PLACE, &unorm, 1, MPIREAL, MPI_SUM, sM->m_comm);
  //MPI_Allreduce(MPI_IN_PLACE, u_avg, 3, MPIREAL, MPI_SUM, sM->m_comm);
  // normalizeFFT is the total number of grid cells
  //u_avg[0] /= sM->normalizeFFT;
  //u_avg[1] /= sM->normalizeFFT;
  //u_avg[2] /= sM->normalizeFFT;
  //unorm = unorm / 2 / sM->normalizeFFT;
  //printf("UNORM %f\n", unorm);
}

void SpectralAnalysis::_fftw2cub() const
{
  const size_t NlocBlocks = sM->local_infos.size();
  const Real * const data_u = sM->data_u;
  const Real * const data_v = sM->data_v;
  const Real * const data_w = sM->data_w;
  const Real factor = 1.0 / (sM->normalizeFFT * 2 * sM->sim.uniformH());

  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<NlocBlocks; ++i) {
    FluidBlock& b = *(FluidBlock*) sM->local_infos[i].ptrBlock;
    const size_t offset = sM->_offset( sM->local_infos[i] );
    for(size_t iz=0; iz< (size_t) FluidBlock::sizeZ; ++iz)
    for(size_t iy=0; iy< (size_t) FluidBlock::sizeY; ++iy)
    for(size_t ix=0; ix< (size_t) FluidBlock::sizeX; ++ix) {
      const size_t src_index = sM->_dest(offset, iz, iy, ix);
      b(ix,iy,iz).tmpU = factor * data_u[src_index];
      b(ix,iy,iz).tmpV = factor * data_v[src_index];
      b(ix,iy,iz).tmpW = factor * data_w[src_index];
    }
  }
}

void SpectralAnalysis::run()
{
  //_cub2fftw();
  //sM->runFwd();
  //sM->_compute_forcing();
  sM->stats.updateDerivedQuantities(sM->sim.nu, sM->sim.dt);
}

void SpectralAnalysis::dump2File() const
{
  if(target not_eq nullptr and target->holdsTargetData
     and sM->sim.time > 5 * target->tInteg) {
    target->updateAvgLogLikelihood(sM->stats,pSamplesCount,avgP,m2P,sM->sim.cs);
    if(sM->sim.time > 100 * target->tInteg) abort(); // end stats collection
  }
  const Real denom =  sM->sim.dt * sM->sim.actualInjectionRate;
  const Real errTKE = (sM->stats.tke - sM->stats.expectedNextTke) / denom;
  if(sM->sim.verbose)
    printf("Re:%e tke:%e errTKE:%e viscousDissip:%e totalDissipRate:%e lIntegral:%e\n",
    sM->stats.Re_lambda, sM->stats.tke, errTKE, sM->stats.dissip_visc,
    sM->stats.dissip_tot, sM->stats.l_integral);

  std::vector<double> buf = std::vector<double>{
    sM->sim.time,         sM->sim.dt,             sM->maxGridL,
    sM->stats.tke,        sM->stats.tke_filtered, sM->stats.dissip_visc,
    sM->stats.dissip_tot, sM->stats.l_integral,   sM->stats.tau_integral,
    sM->sim.nuSgsMean, sM->sim.cs2mean, sM->sim.grad_mean, sM->sim.grad_std
  };

  if(sM->sim.rank==0 and not sM->sim.muteAll) {
    buf.reserve(buf.size() + sM->stats.nBin);
    for (int i=0; i<sM->stats.nBin; ++i) buf.push_back(sM->stats.E_msr[i]);
    FILE * pFile = fopen ("spectralAnalysis.raw", "ab");
    fwrite (buf.data(), sizeof(double), buf.size(), pFile);
    fflush(pFile); fclose(pFile);
    #ifdef ENERGY_FLUX_SPECTRUM
    buf.clear();
    for (int i=0; i<sM->stats.nBin; ++i) buf.push_back(sM->stats.Eflux[i]);
    //buf = std::vector<double>(sM->stats.Eflux, sM->stats.Eflux +sM->stats.nBin);
    pFile = fopen ("fluxAnalysis.raw", "ab");
    fwrite (buf.data(), sizeof(double), buf.size(), pFile);
    fflush(pFile); fclose(pFile);
    #endif
  }

  #if 0
    std::stringstream ssR;
    ssR<<"analysis/spectralAnalysis_"<<std::setfill('0')<<std::setw(9)<<nFile;
    std::ofstream f;
    f.open(ssR.str());
    f <<        "time " << sM->sim.time << "\n";
    f <<          "dt " << sM->sim.dt << "\n";
    f <<        "lBox " << sM->maxGridL << "\n";
    f <<         "tke " << sM->stats.tke << "\n";
    f <<"tke_filtered " << sM->stats.tke_filtered << "\n";
    f << "dissip_visc " << sM->stats.dissip_visc << "\n";
    f <<  "dissip_tot " << sM->stats.dissip_tot << "\n";
    f <<  "l_integral " << sM->stats.l_integral << "\n";
    f <<      "nu_sgs " << sM->sim.nu_sgs << "\n";
    f <<     "cs2_avg " << sM->sim.cs2_avg << "\n";
    f <<   "mean_grad " << sM->sim.grad_mean << "\n";
    f <<    "std_grad " << sM->sim.grad_std << "\n\n";
    f << "k*(lBox/2pi) E_k" << "\n";
    for (int i = 0; i < sM->stats.nBin; ++i)
      f << sM->stats.k_msr[i] << " " <<  << "\n";
    f.flush();
    f.close();
  #endif
  /*
    std::stringstream ssR_cs2;
    ssR_cs2 << "analysis/spectralAnalysisCs2_" << std::setfill('0')
            << std::setw(9) << nFile;
    f.open(ssR_cs2.str());
    f << std::left << "Cs2 spectrum :" << "\n";
    f << std::left << std::setw(15) << "k * (lBox/2pi)"
                   << std::setw(15) << "Cs2_k" << "\n";
    for (int i = 0; i < nBins; i++)
      f << std::left << std::setw(15) << sM->stats.k_msr[i]
                     << std::setw(15) << sM->stats.cs2_msr[i] << "\n";
    f.flush();
    f.close();
  */
}

void SpectralAnalysis::reset()
{
}

SpectralAnalysis::~SpectralAnalysis()
{
  if (target not_eq nullptr) delete target;
}

CubismUP_3D_NAMESPACE_END
