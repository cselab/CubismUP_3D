//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Analysis.h"
#include "../spectralOperators/SpectralAnalysis.h"

#include <sys/stat.h>
#include <iomanip>
#include <sstream>

CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

class KernelAnalysis_gradStats
{
 public:
  Real grad_mean = 0.0;
  Real grad_std  = 0.0;
  const std::array<int, 3> stencil_start = {-1, -1, -1};
  const std::array<int, 3> stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W}};

  KernelAnalysis_gradStats() {}

  ~KernelAnalysis_gradStats() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o)
  {
    const Real h = info.h_gridpoint, fac = 1/(4*h*h);
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);

      const Real d1udx1= LE.u-LW.u, d1vdx1= LE.v-LW.v, d1wdx1= LE.w-LW.w;
      const Real d1udy1= LN.u-LS.u, d1vdy1= LN.v-LS.v, d1wdy1= LN.w-LS.w;
      const Real d1udz1= LB.u-LF.u, d1vdz1= LB.v-LF.v, d1wdz1= LB.w-LF.w;

      const Real grad2 = fac*(d1udx1*d1udx1 + d1vdx1*d1vdx1 + d1wdx1*d1wdx1
                            + d1udy1*d1udy1 + d1vdy1*d1vdy1 + d1wdy1*d1wdy1
                            + d1udz1*d1udz1 + d1vdz1*d1vdz1 + d1wdz1*d1wdz1);
      grad_mean += std::sqrt(grad2);
      grad_std  += grad2;
    }
  }
};

inline void avgUx_wallNormal(Real *avgFlow_xz,
                             const std::vector<BlockInfo>& myInfo,
                             const Real* const uInf, const int bpdy)
{
  size_t nGridPointsY = bpdy * FluidBlock::sizeY;
  const size_t nBlocks = myInfo.size();
  size_t normalize = FluidBlock::sizeX * FluidBlock::sizeZ * nBlocks/bpdy;
  #pragma omp parallel for schedule(static) \
                                      reduction(+ : avgFlow_xz[:2*nGridPointsY])
  for (size_t i = 0; i < nBlocks; i++) {
    const BlockInfo& info = myInfo[i];
    const FluidBlock& b = *(const FluidBlock*)info.ptrBlock;
    // Average Ux on the xz-plane for all y's :
    //     <Ux>_{xz} (y) = Sum_{ix,iz} Ux(ix, y, iz)
    int blockIdxY = info.index[1];
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy) {
      size_t index = blockIdxY * FluidBlock::sizeY + iy;
      //                           sizeY
      //               <----------------------------->
      //
      //               | ... iy-1  iy  iy+1 ...      |
      // ...___,___,___|____,____,____,____,____,____|___,___,___, ...
      //
      //                     blockIdxY                  blockIdxY+1
      for (int ix = 0; ix < FluidBlock::sizeX; ++ix)
      for (int iz = 0; iz < FluidBlock::sizeZ; ++iz) {
        const Real Ux = b(ix, iy, iz).u;
        avgFlow_xz[index                 ] += Ux/normalize;
        avgFlow_xz[index +   nGridPointsY] += Ux*Ux/normalize;
      }
    }
  }
}

Analysis::Analysis(SimulationData& s) : Operator(s) {}

Analysis::~Analysis()
{
  if(sA not_eq nullptr) delete sA;
}

void Analysis::operator()(const double dt)
{
  const bool bFreq = (sim.freqAnalysis>0 && (sim.step+ 1)%sim.freqAnalysis==0);
  const bool bTime = (sim.timeAnalysis>0 && (sim.time+dt)>=sim.nextAnalysisTime);
  const bool bAnalysis =  bFreq || bTime;
  if (not bAnalysis) return;
  sim.nextAnalysisTime += sim.timeAnalysis;

  if (sim.analysis == "channel")
  {
    sim.startProfiler("Channel Analysis");

    int nGridPointsY = sim.bpdy * FluidBlock::sizeY;

    std::vector<Real> avgFlow_xz(2*nGridPointsY, 0);
    avgUx_wallNormal(avgFlow_xz.data(), vInfo, sim.uinf.data(), sim.bpdy);
    MPI_Allreduce(MPI_IN_PLACE, avgFlow_xz.data(), 2*nGridPointsY, MPI_DOUBLE,
                  MPI_SUM, grid->getCartComm());

    // avgFlow_xz = [ <Ux>_{xz}, <Ux^2>_{xz}]
    // Time average, alpha = 1.0 - dt / T_\tau
    // T_\tau = 1/2 L_y / (Re_\tau * <U>)
    // Re_\tau = 180

    /*
    const double T_tau = 0.5 * sim.extent[1] / (180 * sim.uMax_forced);
    const double alpha = (sim.step>0) ? 1.0 - dt / T_tau : 0;
    for (int i = 0; i < nGridPointsY; i++) {
      // Compute <kx>=1/2*(<Ux^2> - <Ux>^2)
      avgFlow_xz[i + nGridPointsY] = 0.5 * (avgFlow_xz[i + nGridPointsY] - avgFlow_xz[i]*avgFlow_xz[i]);
      sim.kx_avg_msr[i] = alpha * sim.kx_avg_msr[i] + (1-alpha) * avgFlow_xz[i + nGridPointsY];
      sim.Ux_avg_msr[i] = alpha * sim.Ux_avg_msr[i] + (1-alpha) * avgFlow_xz[i];
    }
    */

    if (sim.rank==0)
    {
      std::stringstream ssR;
      ssR<<"analysis_"<<std::setfill('0')<<std::setw(9)<<sim.step;
      std::ofstream f;
      f.open (ssR.str());
      f << "Channel Analysis : time=" << sim.time << std::endl;
      f << std::left << std::setw(25) << "<Ux>" << std::setw(25) << "<kx>\n";
      for (int i = 0; i < nGridPointsY; i++) {
        const auto NP = sim.nprocs;
        const Real ux = avgFlow_xz[i]/NP;
        const Real kx = (avgFlow_xz[i+nGridPointsY]/NP - ux*ux)/2;
        f << std::left << std::setw(25) << ux << std::setw(25) << kx << "\n";
      }
    }
    sim.stopProfiler();
    check("Channel Analysis");
  }
  if (sim.analysis == "HIT")
  {
    sim.startProfiler("HIT Analysis");
    //printf("HIT Analysis\n");
    // Compute Gradient stats
    const int nthreads = omp_get_max_threads();
    std::vector<KernelAnalysis_gradStats*> gradStats(nthreads, nullptr);
    #pragma omp parallel for schedule(static, 1)
    for(int i=0; i<nthreads; ++i) gradStats[i] = new KernelAnalysis_gradStats();
    compute<KernelAnalysis_gradStats>(gradStats);

    const size_t normalize = sim.bpdx * FluidBlock::sizeX *
                             sim.bpdy * FluidBlock::sizeY *
                             sim.bpdz * FluidBlock::sizeZ;

    double grad_mean = 0.0, grad_std  = 0.0;
    for (int i=0; i<nthreads; ++i){
      grad_mean += gradStats[i]->grad_mean;
      grad_std  += gradStats[i]->grad_std;
      delete gradStats[i];
    }

    MPI_Allreduce(MPI_IN_PLACE, &grad_mean, 1, MPI_DOUBLE,MPI_SUM,sim.app_comm);
    MPI_Allreduce(MPI_IN_PLACE, &grad_std , 1, MPI_DOUBLE,MPI_SUM,sim.app_comm);
    grad_mean /= normalize;
    grad_std  /= normalize;

    grad_std = std::sqrt(grad_std - grad_mean*grad_mean);

    sim.grad_mean = grad_mean;
    sim.grad_std  = grad_std;

    // Compute spectral analysis
    if(sA == nullptr) sA = new SpectralAnalysis(sim);
    sA->run();
    if (sim.rank==0) sA->dump2File();

    sim.stopProfiler();
    check("HIT Analysis");
  }
}

CubismUP_3D_NAMESPACE_END
