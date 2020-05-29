//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Christian Conti.
//

#ifndef CubismUP_3D_ProcessOperators_h
#define CubismUP_3D_ProcessOperators_h

#include "../SimulationData.h"
#include "../ObstacleBlock.h"
#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

#ifndef CUP_SINGLE_PRECISION
#define MPIREAL MPI_DOUBLE
#else
#define MPIREAL MPI_FLOAT
#endif /* CUP_SINGLE_PRECISION */

inline Real findMaxUzeroMom(const SimulationData& sim)
{
  const std::vector<cubism::BlockInfo>& myInfo = sim.vInfo();
  const Real uinf[3] = {sim.uinf[0], sim.uinf[1], sim.uinf[2]};
  Real mom[3] = {0, 0, 0};
  #pragma omp parallel for schedule(static) reduction(+ : mom[:3])
  for(size_t i=0; i<myInfo.size(); i++)
  {
    const cubism::BlockInfo& info = myInfo[i];
    const FluidBlock& b = *(const FluidBlock *)info.ptrBlock;

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      Real h[3]; info.spacing(h, ix, iy, iz);
      const Real vol = h[0] * h[1] * h[2];
      mom[0] += vol * b(ix,iy,iz).u;
      mom[1] += vol * b(ix,iy,iz).v;
      mom[2] += vol * b(ix,iy,iz).w;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, mom, 3, MPIREAL, MPI_SUM, sim.app_comm);
  const Real corrX = mom[0] / (sim.extent[0] * sim.extent[1] * sim.extent[2]);
  const Real corrY = mom[1] / (sim.extent[0] * sim.extent[1] * sim.extent[2]);
  const Real corrZ = mom[2] / (sim.extent[0] * sim.extent[1] * sim.extent[2]);
  if(sim.verbose)
    printf("Correction in relative momenta:[%e %e %e]\n",corrX,corrY,corrZ);

  Real maxU = 0;
  #pragma omp parallel for schedule(static) reduction(max : maxU)
  for(size_t i=0; i<myInfo.size(); i++)
  {
    const cubism::BlockInfo& info = myInfo[i];
    FluidBlock& b = *(FluidBlock *)info.ptrBlock;

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      b(ix,iy,iz).u -= corrX; const Real u = std::fabs(b(ix,iy,iz).u +uinf[0]);
      b(ix,iy,iz).v -= corrY; const Real v = std::fabs(b(ix,iy,iz).v +uinf[1]);
      b(ix,iy,iz).w -= corrZ; const Real w = std::fabs(b(ix,iy,iz).w +uinf[2]);
      const Real maxUabsAdv = std::max({u, v, w});
      maxU = std::max(maxU, maxUabsAdv);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, & maxU, 1, MPIREAL, MPI_MAX, sim.app_comm);
  assert(maxU >= 0);
  return maxU;
}

inline Real findMaxU(const SimulationData& sim)
{
  const std::vector<cubism::BlockInfo>& myInfo = sim.vInfo();
  const Real uinf[3] = {sim.uinf[0], sim.uinf[1], sim.uinf[2]};

  Real maxU = 0;
  #pragma omp parallel for schedule(static) reduction(max : maxU)
  for(size_t i=0; i<myInfo.size(); i++)
  {
    const cubism::BlockInfo& info = myInfo[i];
    const FluidBlock& b = *(const FluidBlock *)info.ptrBlock;

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const Real advu = std::fabs(b(ix,iy,iz).u + uinf[0]);
      const Real advv = std::fabs(b(ix,iy,iz).v + uinf[1]);
      const Real advw = std::fabs(b(ix,iy,iz).w + uinf[2]);
      const Real maxUl = std::max({advu, advv, advw});
      maxU = std::max(maxU, maxUl);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, & maxU, 1, MPIREAL, MPI_MAX, sim.app_comm);
  assert(maxU >= 0);
  return maxU;
}

#undef MPIREAL

using v_v_ob = std::vector<std::vector<ObstacleBlock*>*>;

inline void putCHIonGrid(
        const std::vector<cubism::BlockInfo>& vInfo,
        const v_v_ob & vec_obstacleBlocks )
{
  #pragma omp parallel for schedule(dynamic,1)
  for(size_t i=0; i<vInfo.size(); i++)
  {
    FluidBlock& b = * (FluidBlock*) vInfo[i].ptrBlock;
    for(int iz=0; iz<FluidBlock::sizeZ; iz++)
    for(int iy=0; iy<FluidBlock::sizeY; iy++)
    for(int ix=0; ix<FluidBlock::sizeX; ix++) b(ix,iy,iz).chi = 0;
    for(size_t o=0; o<vec_obstacleBlocks.size(); o++)
    {
      const auto& pos = ( * vec_obstacleBlocks[o] )[vInfo[i].blockID];
      if(pos == nullptr) continue;
      for(int iz=0; iz<FluidBlock::sizeZ; iz++)
      for(int iy=0; iy<FluidBlock::sizeY; iy++)
      for(int ix=0; ix<FluidBlock::sizeX; ix++)
        b(ix,iy,iz).chi = std::max(pos->chi[iz][iy][ix], b(ix,iy,iz).chi);
    }
  }
}

inline void putSDFonGrid(
        const std::vector<cubism::BlockInfo>& vInfo,
        const v_v_ob & vec_obstacleBlocks )
{
  #pragma omp parallel for schedule(dynamic,1)
  for(size_t i=0; i<vInfo.size(); i++)
  {
    FluidBlock& b = * (FluidBlock*) vInfo[i].ptrBlock;
    for(int iz=0; iz<FluidBlock::sizeZ; iz++)
    for(int iy=0; iy<FluidBlock::sizeY; iy++)
    for(int ix=0; ix<FluidBlock::sizeX; ix++) b(ix,iy,iz).p = -1;
    for(size_t o=0; o<vec_obstacleBlocks.size(); o++)
    {
      const auto& pos = ( * vec_obstacleBlocks[o] )[vInfo[i].blockID];
      if(pos == nullptr) continue;
      for(int iz=0; iz<FluidBlock::sizeZ; iz++)
      for(int iy=0; iy<FluidBlock::sizeY; iy++)
      for(int ix=0; ix<FluidBlock::sizeX; ix++)
        b(ix,iy,iz).p = std::max(pos->sdf[iz][iy][ix], b(ix,iy,iz).p);
    }
  }
}

class KernelVorticity
{
  public:
  KernelVorticity() = default;
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const cubism::StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W}};

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const cubism::BlockInfo& info, BlockType& o) const
  {
    const Real inv2h = .5 / info.h_gridpoint;
    for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for (int iy=0; iy<FluidBlock::sizeY; ++iy)
    for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);
      o(ix,iy,iz).tmpU = inv2h * ( (LN.w-LS.w) - (LB.v-LF.v) );
      o(ix,iy,iz).tmpV = inv2h * ( (LB.u-LF.u) - (LE.w-LW.w) );
      o(ix,iy,iz).tmpW = inv2h * ( (LE.v-LW.v) - (LN.u-LS.u) );
      //o(ix,iy,iz).tmpU =  __FD_2ND(iy, cy, phiS.w, phiC.w, phiN.w)
      //                  - __FD_2ND(iz, cz, phiF.v, phiC.v, phiB.v);
      //o(ix,iy,iz).tmpV =  __FD_2ND(iz, cz, phiF.u, phiC.u, phiB.u)
      //                  - __FD_2ND(ix, cx, phiW.w, phiC.w, phiE.w);
      //o(ix,iy,iz).tmpW =  __FD_2ND(ix, cx, phiW.v, phiC.v, phiE.v)
      //                  - __FD_2ND(iy, cy, phiS.u, phiC.u, phiN.u);
    }
  }
};

class ComputeVorticity : public Operator
{
  public:
  ComputeVorticity(SimulationData & s) : Operator(s) { }
  void operator()(const double dt)
  {
    sim.startProfiler("Vorticity Kernel");
    if(sim.bUseStretchedGrid) {
      printf("TODO Compute Vorticity with stretched grids");
      fflush(0); abort();
    } else {
      const KernelVorticity K;
      compute<KernelVorticity>(K);
    }
    sim.stopProfiler();
    check("Vorticity");
  }
  std::string getName() { return "Vorticity"; }
};

class KernelQcriterion
{
  public:
  KernelQcriterion() = default;
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const cubism::StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W}};

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const cubism::BlockInfo& info, BlockType& o) const
  {
    const Real inv2h = .5 / info.h_gridpoint;
    for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for (int iy=0; iy<FluidBlock::sizeY; ++iy)
    for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);
      const Real WX  = inv2h * ( (LN.w-LS.w) - (LB.v-LF.v) );
      const Real WY  = inv2h * ( (LB.u-LF.u) - (LE.w-LW.w) );
      const Real WZ  = inv2h * ( (LE.v-LW.v) - (LN.u-LS.u) );
      const Real D11 = inv2h * (LE.u-LW.u); // shear stresses
      const Real D22 = inv2h * (LN.v-LS.v); // shear stresses
      const Real D33 = inv2h * (LB.w-LF.w); // shear stresses
      const Real D12 = inv2h * (LN.u-LS.u + LE.v-LW.v); // shear stresses
      const Real D13 = inv2h * (LE.w-LW.w + LB.u-LF.u); // shear stresses
      const Real D23 = inv2h * (LB.v-LF.v + LN.w-LS.w); // shear stresses
      // trace( S S^t ) where S is the sym part of the vel gradient:
      const Real SS = D11*D11 +D22*D22 +D33*D33 +(D12*D12 +D13*D13 +D23*D23)/2;
      o(ix,iy,iz).p = ( (WX*WX + WY*WY + WZ*WZ)/2 - SS ) / 2;
    }
  }
};

class ComputeQcriterion : public Operator
{
  public:
  ComputeQcriterion(SimulationData & s) : Operator(s) { }
  void operator()(const double dt)
  {
    sim.startProfiler("Qcriterion Kernel");
    if(sim.bUseStretchedGrid) {
      printf("TODO Compute Q-criterion with stretched grids");
      fflush(0); abort();
    } else {
      const KernelQcriterion K;
      compute<KernelQcriterion>(K);
    }
    sim.stopProfiler();
    check("Qcriterion");
  }
  std::string getName() { return "Qcriterion"; }
};

#ifdef CUP_ASYNC_DUMP
static void copyDumpGrid(FluidGridMPI& grid, DumpGridMPI& dump)
{
  std::vector<cubism::BlockInfo> vInfo1 = grid.getBlocksInfo();
  std::vector<cubism::BlockInfo> vInfo2 = dump.getBlocksInfo();
  const int N = vInfo1.size();
  if(vInfo1.size() != vInfo2.size()) {
     printf("Async dump fail 1.\n");
     fflush(0); MPI_Abort(grid.getCartComm(), MPI_ERR_OTHER);
   }
  #pragma omp parallel for schedule(static)
  for(int i=0; i<N; i++) {
    const cubism::BlockInfo& info1 = vInfo1[i];
    const cubism::BlockInfo& info2 = vInfo2[i];

    #ifndef NDEBUG
      Real p1[3], p2[3];
      info1.pos(p1, 0,0,0);
      info2.pos(p2, 0,0,0);
      if (fabs(p1[0]-p2[0])>info1.h_gridpoint/2 ||
          fabs(p1[1]-p2[1])>info1.h_gridpoint/2 ||
          fabs(p1[2]-p2[2])>info1.h_gridpoint/2) {
             printf("Async dump fail 2.\n");
             fflush(0); MPI_Abort(grid.getCartComm(), MPI_ERR_OTHER);
          }
    #endif

    const FluidBlock& b = *(FluidBlock*)info1.ptrBlock;
           DumpBlock& d = *( DumpBlock*)info2.ptrBlock;
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      d(ix,iy,iz).u = b(ix,iy,iz).u;
      d(ix,iy,iz).v = b(ix,iy,iz).v;
      d(ix,iy,iz).w = b(ix,iy,iz).w;
      d(ix,iy,iz).chi = b(ix,iy,iz).chi;
      d(ix,iy,iz).p = b(ix,iy,iz).p;
    }
  }
}
#endif

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_ProcessOperators_h
