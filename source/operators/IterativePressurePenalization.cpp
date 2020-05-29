//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "IterativePressurePenalization.h"
#include "../obstacles/ObstacleVector.h"
#ifdef _ACCFFT_
#include "../poisson/PoissonSolverACCPeriodic.h"
#include "../poisson/PoissonSolverACCUnbounded.h"
#else
#include "../poisson/PoissonSolverPeriodic.h"
#include "../poisson/PoissonSolverUnbounded.h"
#endif
// TODO : Cosine transform on GPU!?
#include "../poisson/PoissonSolverMixed.h"
#include "../poisson/PoissonSolverHYPREMixed.h"
#include "../poisson/PoissonSolverPETSCMixed.h"

#include <Cubism/HDF5SliceDumperMPI.h>
#include <iomanip>

// define this to update obstacles with old (mrag-like) approach of integrating
// momenta contained in chi before the penalization step:
#define EXPL_INTEGRATE_MOM

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

using CHIMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE];
using UDEFMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][3];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

namespace
{

static inline PenalizationBlock* getPenalBlockPtr(
  PenalizationGridMPI*const grid, const int blockID) {
  assert(grid not_eq nullptr);
  const std::vector<BlockInfo>& vInfo = grid->getBlocksInfo();
  return (PenalizationBlock*) vInfo[blockID].ptrBlock;
}

struct PressureRHSObstacleVisitor : public ObstacleVisitor
{
  FluidGridMPI * const grid;
  const std::vector<cubism::BlockInfo>& vInfo = grid->getBlocksInfo();

  PressureRHSObstacleVisitor(FluidGridMPI*g) : grid(g) { }

  void visit(Obstacle* const obstacle)
  {
    #pragma omp parallel
    {
      const auto& obstblocks = obstacle->getObstacleBlocks();
      #pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < vInfo.size(); ++i) {
        const cubism::BlockInfo& info = vInfo[i];
        const auto pos = obstblocks[info.blockID];
        if(pos == nullptr) continue;

        FluidBlock& b = *(FluidBlock*)info.ptrBlock;
        const UDEFMAT & __restrict__ UDEF = pos->udef;
        const  CHIMAT & __restrict__  CHI = pos->chi;

        for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          // What if multiple obstacles share a block? Do not write udef onto
          // grid if CHI stored on the grid is greater than obst's CHI.
          if(b(ix,iy,iz).chi > CHI[iz][iy][ix]) continue;
          // What if two obstacles overlap? Let's plus equal. After all here
          // we are computing divUs, maybe one obstacle has divUs 0. We will
          // need a repulsion term of the velocity at some point in the code.
          b(ix,iy,iz).tmpU += UDEF[iz][iy][ix][0];
          b(ix,iy,iz).tmpV += UDEF[iz][iy][ix][1];
          b(ix,iy,iz).tmpW += UDEF[iz][iy][ix][2];
        }
      }
    }
  }
};

struct KernelInitialPressureRHS : public ObstacleVisitor
{
  SimulationData & sim;
  PenalizationGridMPI * const penGrid;
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{
      -1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W,FE_TMPU,FE_TMPV,FE_TMPW}};
  ObstacleVector * const obstacle_vector = sim.obstacle_vector;
  const int nShapes = obstacle_vector->nObstacles();
  // non-const non thread safe:
  std::vector<Real> sumRHS = std::vector<Real>(nShapes, 0);
  std::vector<Real> supRHS = std::vector<Real>(nShapes, 0);
  // modified before going into accept
  const cubism::BlockInfo * info_ptr = nullptr;
  PenalizationBlock * pblock_ptr = nullptr;
  Lab * lab_ptr = nullptr;

  KernelInitialPressureRHS(SimulationData& s, PenalizationGridMPI * const pen)
    : sim(s), penGrid(pen) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & l, const BlockInfo& info, BlockType& o)
  {
    PenalizationBlock& t = * getPenalBlockPtr(penGrid, info.blockID);
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement & LW = l(ix-1,iy,  iz  ), & LE = l(ix+1,iy,  iz  );
      const FluidElement & LS = l(ix,  iy-1,iz  ), & LN = l(ix,  iy+1,iz  );
      const FluidElement & LF = l(ix,  iy,  iz-1), & LB = l(ix,  iy,  iz+1);
      t(ix,iy,iz).rhs0 = LE.u   -LW.u    + LN.v   -LS.v    + LB.w   -LF.w;
    }

    // first store the lab and info, then do visitor
    assert(info_ptr == nullptr && lab_ptr == nullptr && pblock_ptr == nullptr);
    info_ptr = & info; lab_ptr = & l; pblock_ptr = & t;
    ObstacleVisitor* const base = static_cast<ObstacleVisitor*> (this);
    assert( base not_eq nullptr );
    obstacle_vector->Accept( base );
    info_ptr = nullptr; lab_ptr = nullptr; pblock_ptr = nullptr;
  }

  void visit(Obstacle* const obstacle)
  {
    assert(info_ptr not_eq nullptr && lab_ptr not_eq nullptr);
    const BlockInfo & info = * info_ptr;
    const auto& obstblocks = obstacle->getObstacleBlocks();
    if (obstblocks[info.blockID] == nullptr) return;
    PenalizationBlock & t = * pblock_ptr;
    Lab & lab = * lab_ptr;
    const CHIMAT & __restrict__ CHI = obstblocks[info.blockID]->chi;
    const int obstID = obstacle->obstacleID; assert(obstID < nShapes);

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      if (lab(ix,iy,iz).chi > CHI[iz][iy][ix]) continue;
      const FluidElement &LW = lab(ix-1,iy,  iz  ), &LE = lab(ix+1,iy,  iz  );
      const FluidElement &LS = lab(ix,  iy-1,iz  ), &LN = lab(ix,  iy+1,iz  );
      const FluidElement &LF = lab(ix,  iy,  iz-1), &LB = lab(ix,  iy,  iz+1);
      const Real divUs = LE.tmpU-LW.tmpU + LN.tmpV-LS.tmpV + LB.tmpW-LF.tmpW;
      #ifdef EXPL_INTEGRATE_MOM
        const Real srcBulk = - CHI[iz][iy][ix] * divUs;
      #else
        const Real XlamDt = sim.lambda * sim.dt * CHI[iz][iy][ix];
        const Real srcBulk = - XlamDt/(1 + XlamDt) * divUs;
      #endif
      t(ix,iy,iz).rhs0 += srcBulk;
      sumRHS[obstID] += srcBulk;
      supRHS[obstID] += CHI[iz][iy][ix];
    }
  }
};

// place in PenalizationBlocks velocity before iterations plus pressure correct
struct KernelIterateGradP
{
  const Real dt;
  PenalizationGridMPI * const penGrid;
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_P}}};

  KernelIterateGradP(double _dt, PenalizationGridMPI * const pen) :
    dt(_dt), penGrid{pen} {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & l, const BlockInfo& info, BlockType& o) const
  {
    const Real fac = - 0.5 * dt / info.h_gridpoint;
    PenalizationBlock& t = * getPenalBlockPtr(penGrid, info.blockID);
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      t(ix,iy,iz).uPres = l(ix,iy,iz).u + fac*(l(ix+1,iy,iz).p-l(ix-1,iy,iz).p);
      t(ix,iy,iz).vPres = l(ix,iy,iz).v + fac*(l(ix,iy+1,iz).p-l(ix,iy-1,iz).p);
      t(ix,iy,iz).wPres = l(ix,iy,iz).w + fac*(l(ix,iy,iz+1).p-l(ix,iy,iz-1).p);
    }
  }
};

struct KernelIntegrateFluidMomenta : public ObstacleVisitor
{
  SimulationData & sim;
  PenalizationGridMPI * const penGrid;
  const cubism::BlockInfo * info_ptr = nullptr;
  inline double dvol(const cubism::BlockInfo&info, const int x, const int y, const int z) const {
    double h[3]; info.spacing(h, x, y, z);
    return h[0] * h[1] * h[2];
  }

  KernelIntegrateFluidMomenta(SimulationData& s, PenalizationGridMPI*const pen)
    : sim(s), penGrid(pen) {}

  void operator()(const cubism::BlockInfo& info)
  {
    // first store the lab and info, then do visitor
    assert(info_ptr == nullptr);
    info_ptr = & info;
    ObstacleVisitor* const base = static_cast<ObstacleVisitor*> (this);
    assert( base not_eq nullptr );
    sim.obstacle_vector->Accept( base );
    info_ptr = nullptr;
  }

  void visit(Obstacle* const op)
  {
    const BlockInfo& info = * info_ptr;
    assert(info_ptr not_eq nullptr);
    const std::vector<ObstacleBlock*>& obstblocks = op->getObstacleBlocks();
    ObstacleBlock*const o = obstblocks[info.blockID];
    if (o == nullptr) return;

    const std::array<double,3> CM = op->getCenterOfMass();
    const PenalizationBlock& b = * getPenalBlockPtr(penGrid, info.blockID);

    const CHIMAT & __restrict__ CHI = o->chi;
    double &VV = o->V;
    double &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    double &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    VV = 0; FX = 0; FY = 0; FZ = 0; TX = 0; TY = 0; TZ = 0;
    double &J0 = o->J0, &J1 = o->J1, &J2 = o->J2;
    double &J3 = o->J3, &J4 = o->J4, &J5 = o->J5;
    J0 = 0; J1 = 0; J2 = 0; J3 = 0; J4 = 0; J5 = 0;

    #ifndef EXPL_INTEGRATE_MOM
      const UDEFMAT & __restrict__ UDEF = o->udef;
      o->GfX = 0;
      o->GpX = 0; o->GpY = 0; o->GpZ = 0;
      o->Gj0 = 0; o->Gj1 = 0; o->Gj2 = 0;
      o->Gj3 = 0; o->Gj4 = 0; o->Gj5 = 0;
      o->GuX = 0; o->GuY = 0; o->GuZ = 0;
      o->GaX = 0; o->GaY = 0; o->GaZ = 0;
      const Real lambdt = sim.lambda * sim.dt;
    #endif

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      if (CHI[iz][iy][ix] <= 0) continue;
      double p[3]; info.pos(p, ix, iy, iz);
      const double dv = dvol(info, ix, iy, iz), X = CHI[iz][iy][ix];
      p[0] -= CM[0]; p[1] -= CM[1]; p[2] -= CM[2];

      VV += X * dv;
      J0 += X * dv * ( p[1]*p[1] + p[2]*p[2] );
      J1 += X * dv * ( p[0]*p[0] + p[2]*p[2] );
      J2 += X * dv * ( p[0]*p[0] + p[1]*p[1] );
      J3 -= X * dv * p[0]*p[1];
      J4 -= X * dv * p[0]*p[2];
      J5 -= X * dv * p[1]*p[2];

      FX += X * dv * b(ix,iy,iz).uPres;
      FY += X * dv * b(ix,iy,iz).vPres;
      FZ += X * dv * b(ix,iy,iz).wPres;
      TX += X * dv * ( p[1] * b(ix,iy,iz).wPres - p[2] * b(ix,iy,iz).vPres );
      TY += X * dv * ( p[2] * b(ix,iy,iz).uPres - p[0] * b(ix,iy,iz).wPres );
      TZ += X * dv * ( p[0] * b(ix,iy,iz).vPres - p[1] * b(ix,iy,iz).uPres );

      #ifndef EXPL_INTEGRATE_MOM
        const Real penalFac = dv * lambdt * X / ( 1 + X * lambdt );
        o->GfX += penalFac;
        o->GpX += penalFac * p[0];
        o->GpY += penalFac * p[1];
        o->GpZ += penalFac * p[2];
        o->Gj0 += penalFac * ( p[1]*p[1] + p[2]*p[2] );
        o->Gj1 += penalFac * ( p[0]*p[0] + p[2]*p[2] );
        o->Gj2 += penalFac * ( p[0]*p[0] + p[1]*p[1] );
        o->Gj3 -= penalFac * p[0]*p[1];
        o->Gj4 -= penalFac * p[0]*p[2];
        o->Gj5 -= penalFac * p[1]*p[2];
        const double DiffU[3] = {
          b(ix,iy,iz).uPres - UDEF[iz][iy][ix][0],
          b(ix,iy,iz).vPres - UDEF[iz][iy][ix][1],
          b(ix,iy,iz).wPres - UDEF[iz][iy][ix][2]
        };
        o->GuX += penalFac * DiffU[0];
        o->GuY += penalFac * DiffU[1];
        o->GuZ += penalFac * DiffU[2];
        o->GaX += penalFac * ( p[1] * DiffU[2] - p[2] * DiffU[1] );
        o->GaY += penalFac * ( p[2] * DiffU[0] - p[0] * DiffU[2] );
        o->GaZ += penalFac * ( p[0] * DiffU[1] - p[1] * DiffU[0] );
      #endif
    }
  }
};

struct KernelFinalizeObstacleVel : public ObstacleVisitor
{
  FluidGridMPI * const grid;
  KernelFinalizeObstacleVel(FluidGridMPI*g) : grid(g) { }

  void visit(Obstacle* const obst)
  {
    static constexpr int nQoI = 29;
    double M[nQoI] = { 0 };
    const auto& oBlock = obst->getObstacleBlocks();
    #pragma omp parallel for schedule(static,1) reduction(+ : M[:nQoI])
    for (size_t i=0; i<oBlock.size(); i++)
    {
      if(oBlock[i] == nullptr) continue;
      M[ 0] += oBlock[i]->V ;
      M[ 1] += oBlock[i]->FX; M[ 2] += oBlock[i]->FY; M[ 3] += oBlock[i]->FZ;
      M[ 4] += oBlock[i]->TX; M[ 5] += oBlock[i]->TY; M[ 6] += oBlock[i]->TZ;
      M[ 7] += oBlock[i]->J0; M[ 8] += oBlock[i]->J1; M[ 9] += oBlock[i]->J2;
      M[10] += oBlock[i]->J3; M[11] += oBlock[i]->J4; M[12] += oBlock[i]->J5;
      #ifndef EXPL_INTEGRATE_MOM
        M[13]+= oBlock[i]->GfX;
        M[14]+= oBlock[i]->GpX; M[15]+= oBlock[i]->GpY; M[16]+= oBlock[i]->GpZ;
        M[17]+= oBlock[i]->Gj0; M[18]+= oBlock[i]->Gj1; M[19]+= oBlock[i]->Gj2;
        M[20]+= oBlock[i]->Gj3; M[21]+= oBlock[i]->Gj4; M[22]+= oBlock[i]->Gj5;
        M[23]+= oBlock[i]->GuX; M[24]+= oBlock[i]->GuY; M[25]+= oBlock[i]->GuZ;
        M[26]+= oBlock[i]->GaX; M[27]+= oBlock[i]->GaY; M[28]+= oBlock[i]->GaZ;
      #endif
    }
    const auto comm = grid->getCartComm();
    MPI_Allreduce(MPI_IN_PLACE, M, nQoI, MPI_DOUBLE, MPI_SUM, comm);
    assert(std::fabs(obst->mass - M[ 0]) < 10*EPS);
    assert(std::fabs(obst->J[0] - M[ 7]) < 10*EPS);
    assert(std::fabs(obst->J[1] - M[ 8]) < 10*EPS);
    assert(std::fabs(obst->J[2] - M[ 9]) < 10*EPS);
    assert(std::fabs(obst->J[3] - M[10]) < 10*EPS);
    assert(std::fabs(obst->J[4] - M[11]) < 10*EPS);
    assert(std::fabs(obst->J[5] - M[12]) < 10*EPS);
    assert(M[0] > EPS);

    #ifndef EXPL_INTEGRATE_MOM
      obst->penalM    = M[13];
      obst->penalCM   = { M[14], M[15], M[16] };
      obst->penalJ    = { M[17], M[18], M[19], M[20], M[21], M[22] };
      obst->penalLmom = { M[23], M[24], M[25] };
      obst->penalAmom = { M[26], M[27], M[28] };
    #else
      obst->penalM    = M[0];
      obst->penalCM   = { 0, 0, 0 };
      obst->penalJ    = { M[ 7], M[ 8], M[ 9], M[10], M[11], M[12] };
      obst->penalLmom = { M[1], M[2], M[3] };
      obst->penalAmom = { M[4], M[5], M[6] };
    #endif

    obst->computeVelocities();
  }
};

struct KernelFinalizePenalizationForce : public ObstacleVisitor
{
  FluidGridMPI * const grid;
  KernelFinalizePenalizationForce(FluidGridMPI*g) : grid(g) { }

  void visit(Obstacle* const obst)
  {
    static constexpr int nQoI = 6;
    double M[nQoI] = { 0 };
    const auto& oBlock = obst->getObstacleBlocks();
    #pragma omp parallel for schedule(static) reduction(+ : M[:nQoI])
    for (size_t i=0; i<oBlock.size(); ++i) {
      if(oBlock[i] == nullptr) continue;
      M[0] += oBlock[i]->FX; M[1] += oBlock[i]->FY; M[2] += oBlock[i]->FZ;
      M[3] += oBlock[i]->TX; M[4] += oBlock[i]->TY; M[5] += oBlock[i]->TZ;
    }
    MPI_Allreduce(MPI_IN_PLACE, M,nQoI, MPI_DOUBLE, MPI_SUM, grid->getCartComm());
    obst->force[0]  = M[0];
    obst->force[1]  = M[1];
    obst->force[2]  = M[2];
    obst->torque[0] = M[3];
    obst->torque[1] = M[4];
    obst->torque[2] = M[5];
  }
};

struct KernelPenalization : public ObstacleVisitor
{
  SimulationData & sim;
  const double lambda = sim.lambda, dt = sim.dt;
  PenalizationGridMPI * const penGrid;
  Real MX = 0, MY = 0, MZ = 0, DMX = 0, DMY = 0, DMZ = 0;
  const cubism::BlockInfo * info_ptr = nullptr;

  KernelPenalization(SimulationData& s, PenalizationGridMPI* _pen)
    : sim(s), penGrid(_pen) {}

  void operator()(const cubism::BlockInfo& info)
  {
    // first store the lab and info, then do visitor
    info_ptr = & info;
    ObstacleVisitor* const base = static_cast<ObstacleVisitor*> (this);
    assert( base not_eq nullptr );
    sim.obstacle_vector->Accept( base );
    info_ptr = nullptr;
  }

  void visit(Obstacle* const obstacle)
  {
    const BlockInfo& info = * info_ptr;
    assert(info_ptr not_eq nullptr);
    const auto& obstblocks = obstacle->getObstacleBlocks();
    ObstacleBlock*const o = obstblocks[info.blockID];
    if (o == nullptr) return;

    const CHIMAT & __restrict__ CHI = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
    FluidBlock& b = *(FluidBlock*)info.ptrBlock;
    PenalizationBlock& t = * getPenalBlockPtr(penGrid, info.blockID);

    double &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    double &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    FX = 0; FY = 0; FZ = 0; TX = 0; TY = 0; TZ = 0;

    const std::array<double,3> CM = obstacle->getCenterOfMass();
    const std::array<double,3> vel = obstacle->getTranslationVelocity();
    const std::array<double,3> omega = obstacle->getAngularVelocity();
    const double dv = std::pow(info.h_gridpoint, 3);

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(b(ix,iy,iz).chi > CHI[iz][iy][ix]) continue;

      Real p[3]; info.pos(p, ix, iy, iz);
      p[0] -= CM[0]; p[1] -= CM[1]; p[2] -= CM[2];
      const Real X = CHI[iz][iy][ix], U_TOT[3] = {
          (Real)(vel[0] + omega[1]*p[2] - omega[2]*p[1] + UDEF[iz][iy][ix][0]),
          (Real)(vel[1] + omega[2]*p[0] - omega[0]*p[2] + UDEF[iz][iy][ix][1]),
          (Real)(vel[2] + omega[0]*p[1] - omega[1]*p[0] + UDEF[iz][iy][ix][2])
      };
      #ifdef EXPL_INTEGRATE_MOM
        const Real penalFac = X * lambda;
      #else
        const Real penalFac = X * lambda / (1 + dt * X * lambda);
      #endif
      const Real FPX = penalFac * (U_TOT[0] - t(ix,iy,iz).uPres);
      const Real FPY = penalFac * (U_TOT[1] - t(ix,iy,iz).vPres);
      const Real FPZ = penalFac * (U_TOT[2] - t(ix,iy,iz).wPres);
      const Real oldF[]= {b(ix,iy,iz).tmpU, b(ix,iy,iz).tmpV, b(ix,iy,iz).tmpW};
      b(ix,iy,iz).tmpU = dt * FPX;
      b(ix,iy,iz).tmpV = dt * FPY;
      b(ix,iy,iz).tmpW = dt * FPZ;

      FX += dv * FPX; FY += dv * FPY; FZ += dv * FPZ;
      TX += dv * ( p[1] * FPZ - p[2] * FPY );
      TY += dv * ( p[2] * FPX - p[0] * FPZ );
      TZ += dv * ( p[0] * FPY - p[1] * FPX );
      MX  += std::pow(b(ix,iy,iz).tmpU        , 2);
      DMX += std::pow(b(ix,iy,iz).tmpU-oldF[0], 2);
      MY  += std::pow(b(ix,iy,iz).tmpV        , 2);
      DMY += std::pow(b(ix,iy,iz).tmpV-oldF[1], 2);
      MZ  += std::pow(b(ix,iy,iz).tmpW        , 2);
      DMZ += std::pow(b(ix,iy,iz).tmpW-oldF[2], 2);
    }
  }
};

struct KernelPressureRHS
{
  SimulationData & sim;
  PenalizationGridMPI * const penGrid;
  PoissonSolver * const solver = sim.pressureSolver;
  const int nShapes = sim.obstacle_vector->nObstacles();
  // non-const non thread safe:
  std::vector<Real> penRHS = std::vector<Real>(nShapes, 0);

  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_TMPU,FE_TMPV,FE_TMPW}};

  KernelPressureRHS(SimulationData& s, PenalizationGridMPI*const pen)
    : sim(s), penGrid(pen) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o)
  {
    const Real h = info.h_gridpoint, fac = h * h / sim.dt / 2;
    const PenalizationBlock& t = * getPenalBlockPtr(penGrid, info.blockID);
    Real* __restrict__ const ret = solver->data + solver->_offset_ext(info);
    const unsigned SX=solver->stridex, SY=solver->stridey, SZ=solver->stridez;

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const PenalizationHelperElement& P = t(ix,iy,iz);
      const FluidElement &LW = lab(ix-1,iy,  iz  ), &LE = lab(ix+1,iy,  iz  );
      const FluidElement &LS = lab(ix,  iy-1,iz  ), &LN = lab(ix,  iy+1,iz  );
      const FluidElement &LF = lab(ix,  iy,  iz-1), &LB = lab(ix,  iy,  iz+1);
      const Real divFdt = LE.tmpU-LW.tmpU + LN.tmpV-LS.tmpV + LB.tmpW-LF.tmpW;
      ret[SZ*iz + SY*iy + SX*ix] = fac * (P.rhs0 + divFdt);
      penRHS[0] += divFdt;
    }
  }
};

// place on grid the penalized velocity plus pressure correction
struct KernelGradP
{
  const Real dt;
  PenalizationGridMPI * const penGrid;
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_P}}};

  KernelGradP(double _dt, PenalizationGridMPI*const pen): dt(_dt),penGrid{pen}{}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const Real fac = - 0.5 * dt / info.h_gridpoint;
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const Real dUpres = fac * (lab(ix+1,iy,iz).p - lab(ix-1,iy,iz).p);
      const Real dVpres = fac * (lab(ix,iy+1,iz).p - lab(ix,iy-1,iz).p);
      const Real dWpres = fac * (lab(ix,iy,iz+1).p - lab(ix,iy,iz-1).p);
      o(ix,iy,iz).u = o(ix,iy,iz).u + o(ix,iy,iz).tmpU + dUpres;
      o(ix,iy,iz).v = o(ix,iy,iz).v + o(ix,iy,iz).tmpV + dVpres;
      o(ix,iy,iz).w = o(ix,iy,iz).w + o(ix,iy,iz).tmpW + dWpres;
    }
  }
};

struct KernelFinalizePerimeters : public ObstacleVisitor
{
  SimulationData & sim;
  const std::vector<cubism::BlockInfo>& vInfo = sim.grid->getBlocksInfo();
  PoissonSolver * const solver = sim.pressureSolver;
  const std::vector<Real> & corrFac;

  KernelFinalizePerimeters(SimulationData & s, const std::vector<Real> & corr)
    : sim(s), corrFac(corr) {}

  void visit(Obstacle* const obstacle)
  {
    const unsigned SX=solver->stridex, SY=solver->stridey, SZ=solver->stridez;
    const int obstID = obstacle->obstacleID;

    #pragma omp parallel
    {
      const auto& obstblocks = obstacle->getObstacleBlocks();
      #pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < vInfo.size(); ++i)
      {
        const cubism::BlockInfo& info = vInfo[i];
        if(obstblocks[info.blockID] == nullptr) continue;

        const CHIMAT & __restrict__ CHI = obstblocks[info.blockID]->chi;
        const Real h = info.h_gridpoint, fac = corrFac[obstID] * h*h/sim.dt/2;
        const size_t offset = solver->_offset_ext(info);
        Real* __restrict__ const ret = solver->data;

        for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
          ret[offset + SZ * iz + SY * iy + SX * ix] -= fac * CHI[iz][iy][ix];
      }
    }
  }
};

}

void IterativePressurePenalization::initializeFields()
{
}

IterativePressurePenalization::IterativePressurePenalization(SimulationData& s)
  : Operator(s)
{
  if(sim.bUseFourierBC)
    pressureSolver = new PoissonSolverPeriodic(sim);
  else if (sim.bUseUnboundedBC)
    pressureSolver = new PoissonSolverUnbounded(sim);
  #ifdef CUP_HYPRE
  else if (sim.useSolver == "hypre")
    pressureSolver = new PoissonSolverMixed_HYPRE(sim);
  #endif
  #ifdef CUP_PETSC
  else if (sim.useSolver == "petsc")
    pressureSolver = new PoissonSolverMixed_PETSC(sim);
  #endif
  else
    pressureSolver = new PoissonSolverMixed(sim);
    //pressureSolver = new PoissonSolverPeriodic(sim);
  sim.pressureSolver = pressureSolver;

  // no need to allocate stretched mesh stuff here because we will never read
  // grid spacing from this grid!
  if(sim.rank==0) printf("Allocating the penalization helper grid.\n");

  penalizationGrid = new PenalizationGridMPI(
    sim.nprocsx, sim.nprocsy, sim.nprocsz, sim.local_bpdx,
    sim.local_bpdy, sim.local_bpdz, sim.maxextent, sim.app_comm);
}

void IterativePressurePenalization::operator()(const double dt)
{
  const size_t nShapes = sim.obstacle_vector->nObstacles();
  const int nthreads = omp_get_max_threads();
  std::vector<double> sumDEF(nShapes, 0), supRHS(nShapes, 0);
  #ifdef EXPL_INTEGRATE_MOM
    sim.lambda = 1.0/dt;
  #endif
  // first copy velocity before either Pres or Penal onto penalization blocks
  // also put udef into tmpU fields
  { //zero fields, going to contain Udef:
    sim.startProfiler("PresRHS Udef");
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      assert((size_t) vInfo[i].blockID == i);
      FluidBlock & b = * (FluidBlock*) vInfo[i].ptrBlock;
      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy,iz).tmpU = 0; b(ix,iy,iz).tmpV = 0; b(ix,iy,iz).tmpW = 0;
      }
    }
    //store deformation velocities onto tmp fields:
    ObstacleVisitor* visitor = new PressureRHSObstacleVisitor(grid);
    sim.obstacle_vector->Accept(visitor);
    delete visitor;
    sim.stopProfiler();
  }
  {
    // compute velocity after pressure projection (PP) but without Penal
    // using pressure computed at prior step / iteration
    sim.startProfiler("rhs0"); //pressure correction dudt* = - grad P / rho
    std::vector<KernelInitialPressureRHS*> K(nthreads, nullptr);
    for(int i=0; i<nthreads; ++i)
      K[i] = new KernelInitialPressureRHS (sim, penalizationGrid);
    compute<KernelInitialPressureRHS>(K);
    for(size_t j = 0; j<nShapes; ++j) {
      for(int i=0; i<nthreads; ++i) sumDEF[j] += K[i]->sumRHS[j];
      for(int i=0; i<nthreads; ++i) supRHS[j] += K[i]->supRHS[j];
    }
    for(int i=0; i<nthreads; i++) delete K[i];

    const auto& COMM = sim.app_comm;
    MPI_Allreduce(MPI_IN_PLACE, sumDEF.data(), nShapes, MPI_DOUBLE,MPI_SUM, COMM);
    MPI_Allreduce(MPI_IN_PLACE, supRHS.data(), nShapes, MPI_DOUBLE,MPI_SUM, COMM);

    sim.stopProfiler();
  }

  int iter=0;
  Real relDF = 1e3;
  bool bConverged = false;
  for(iter = 0; iter < 1000; ++iter)
  {
    {
      // compute velocity after pressure projection (PP) but without Penal
      // using pressure computed at prior step / iteration
      sim.startProfiler("GradP"); //pressure correction dudt* = - grad P / rho
      const KernelIterateGradP K(dt, penalizationGrid);
      compute<KernelIterateGradP>(K);
      sim.stopProfiler();
    }

    { // integrate momenta by looping over grid
      sim.startProfiler("Obst Int Vel");
      #pragma omp parallel
      { // each thread needs to call its own non-const operator() function
        KernelIntegrateFluidMomenta K(sim, penalizationGrid);
        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
      }

      ObstacleVisitor*K = new KernelFinalizeObstacleVel(sim.grid);
      sim.obstacle_vector->Accept(K); // accept you son of a french cow
      delete K;
      sim.stopProfiler();
    }

    // finally update vel with penalization but without pressure
    {
      sim.startProfiler("Penalization");
      double M[6] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
      #pragma omp parallel reduction (+ : M[:6])
      { // each thread needs to call its own non-const operator() function
        KernelPenalization K(sim, penalizationGrid);
        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
        M[0] += K.MX; M[3] += K.DMX;
        M[1] += K.MY; M[4] += K.DMY;
        M[2] += K.MZ; M[5] += K.DMZ;
      }
      const auto comm = sim.grid->getCartComm();
      MPI_Allreduce(MPI_IN_PLACE, M, 6, MPI_DOUBLE, MPI_SUM, comm);
      relDF = std::sqrt( (M[3]+M[4]+M[5]) / (EPS + M[0]+M[1]+M[2]) );

      ObstacleVisitor*K = new KernelFinalizePenalizationForce(sim.grid);
      sim.obstacle_vector->Accept(K); // accept you son of a french cow
      delete K;

      sim.stopProfiler();
    }

    if(sim.verbose) printf("iter:%02d - max relative error: %f\n", iter, relDF);
    //if(relDF < 0.001) break;
    if(iter>0 && relDF<0.002) bConverged = true;

    {
      sim.startProfiler("PresRHS Kernel");
      //place onto p: ( div u^(t+1) - div u^* ) / dt
      //where i want div u^(t+1) to be equal to div udef
      sim.pressureSolver->reset();
      std::vector<KernelPressureRHS*> K(nthreads, nullptr);
      for(int i=0; i<nthreads; ++i)
          K[i] = new KernelPressureRHS (sim, penalizationGrid);
      compute<KernelPressureRHS>(K);
      std::vector<double> sumPEN(nShapes, 0);
      for(size_t j=0; j<nShapes; ++j)
        for(int i=0; i<nthreads; ++i) sumPEN[j] += K[i]->penRHS[j];
      for(int i=0; i<nthreads; i++) delete K[i];

      const auto& COMM = sim.app_comm;
      MPI_Allreduce(MPI_IN_PLACE, sumPEN.data(), nShapes, MPI_DOUBLE,MPI_SUM, COMM);
      std::vector<Real> corrFactors(nShapes, 0);
      for(size_t j = 0; j<nShapes; ++j)
        corrFactors[j] = (sumDEF[j]+sumPEN[j])/std::max(supRHS[j],(double)EPS);
      sim.stopProfiler();

      sim.startProfiler("PresRHS Correct");
      ObstacleVisitor * K2 = new KernelFinalizePerimeters(sim, corrFactors);
      sim.obstacle_vector->Accept(K2); // accept you son of a french cow
      delete K2;
      sim.stopProfiler();
    }

    pressureSolver->solve();

    sim.startProfiler("sol2cub");
    pressureSolver->_fftw2cub();
    sim.stopProfiler();

    if(bConverged) break;
  }

  sim.startProfiler("GradP"); //pressure correction dudt* = - grad P / rho
  {
    const KernelGradP K(dt, penalizationGrid);
    compute<KernelGradP>(K);
  }
  sim.stopProfiler();

  /*
  if(not sim.muteAll)
  {
  std::stringstream ssF; ssF<<sim.path2file<<"/pressureIterStats.dat";
  std::ofstream pfile(ssF.str().c_str(), std::ofstream::app);
  if(sim.step==0) pfile<<"step time dt iter relDF"<<std::endl;
  pfile<<sim.step<<" "<<sim.time<<" "<<sim.dt<<" "<<iter<<" "<<relDF<<std::endl;
  }
  */

  check("IterativePressurePenalization");
}

CubismUP_3D_NAMESPACE_END
