//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Penalization.h"
#include "../obstacles/ObstacleVector.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

using CHIMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE];
using UDEFMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][3];

template<bool implicitPenalization>
struct KernelPenalization : public ObstacleVisitor
{
  const Real dt, invdt = 1.0/dt, lambda;
  ObstacleVector * const obstacle_vector;
  const cubism::BlockInfo * info_ptr = nullptr;

  KernelPenalization(double _dt, double _lambda, ObstacleVector* ov) :
    dt(_dt), lambda(_lambda), obstacle_vector(ov) {}

  void operator()(const cubism::BlockInfo& info)
  {
    // first store the lab and info, then do visitor
    info_ptr = & info;
    ObstacleVisitor* const base = static_cast<ObstacleVisitor*> (this);
    assert( base not_eq nullptr );
    obstacle_vector->Accept( base );
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
    const std::array<double,3> CM = obstacle->getCenterOfMass();
    const std::array<double,3> vel = obstacle->getTranslationVelocity();
    const std::array<double,3> omega = obstacle->getAngularVelocity();
    const Real dv = std::pow(info.h_gridpoint, 3);

    // Obstacle-specific lambda, useful for gradually adding an obstacle to the flow.
    const double rampUp = obstacle->lambda_factor;
    // lambda = 1/dt hardcoded for expl time int, other options are wrong.
    const double lambdaFac = rampUp * (implicitPenalization? lambda : invdt);

    double &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    double &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    FX = 0; FY = 0; FZ = 0; TX = 0; TY = 0; TZ = 0;

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      // What if multiple obstacles share a block? Do not write udef onto
      // grid if CHI stored on the grid is greater than obst's CHI.
      if(b(ix,iy,iz).chi > CHI[iz][iy][ix]) continue;
      if(CHI[iz][iy][ix] <= 0) continue; // no need to do anything
      double p[3]; info.pos(p, ix, iy, iz);
      p[0] -= CM[0]; p[1] -= CM[1]; p[2] -= CM[2];

      const double X = CHI[iz][iy][ix], U_TOT[3] = {
          vel[0] + omega[1]*p[2] - omega[2]*p[1] + UDEF[iz][iy][ix][0],
          vel[1] + omega[2]*p[0] - omega[0]*p[2] + UDEF[iz][iy][ix][1],
          vel[2] + omega[0]*p[1] - omega[1]*p[0] + UDEF[iz][iy][ix][2]
      };
      const Real penalFac = implicitPenalization? X*lambdaFac/(1+lambdaFac*dt*X)
                                                : X*lambdaFac;

      const Real FPX = penalFac * (U_TOT[0] - b(ix,iy,iz).u);
      const Real FPY = penalFac * (U_TOT[1] - b(ix,iy,iz).v);
      const Real FPZ = penalFac * (U_TOT[2] - b(ix,iy,iz).w);
      // What if two obstacles overlap? Let's plus equal. We will need a
      // repulsion term of the velocity at some point in the code.
      b(ix,iy,iz).u = b(ix,iy,iz).u + dt * FPX;
      b(ix,iy,iz).v = b(ix,iy,iz).v + dt * FPY;
      b(ix,iy,iz).w = b(ix,iy,iz).w + dt * FPZ;

      FX += dv * FPX; FY += dv * FPY; FZ += dv * FPZ;
      TX += dv * ( p[1] * FPZ - p[2] * FPY );
      TY += dv * ( p[2] * FPX - p[0] * FPZ );
      TZ += dv * ( p[0] * FPY - p[1] * FPX );
    }
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
    const auto comm = grid->getCartComm();
    MPI_Allreduce(MPI_IN_PLACE, M, nQoI, MPI_DOUBLE, MPI_SUM, comm);
    obst->force[0]  = M[0]; obst->force[1]  = M[1]; obst->force[2]  = M[2];
    obst->torque[0] = M[3]; obst->torque[1] = M[4]; obst->torque[2] = M[5];
  }
};

}

Penalization::Penalization(SimulationData & s) : Operator(s) {}

void Penalization::operator()(const double dt)
{
  if(sim.obstacle_vector->nObstacles() == 0) return;

  sim.startProfiler("Penalization");
  #pragma omp parallel
  { // each thread needs to call its own non-const operator() function
    if(sim.bImplicitPenalization)
    {
      KernelPenalization<1> K(dt, sim.lambda, sim.obstacle_vector);
      #pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
    }
    else
    {
      KernelPenalization<0> K(dt, sim.lambda, sim.obstacle_vector);
      #pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
    }
  }

  ObstacleVisitor*K = new KernelFinalizePenalizationForce(sim.grid);
  sim.obstacle_vector->Accept(K); // accept you son of a french cow
  delete K;

  sim.stopProfiler();
  check("Penalization");
}

CubismUP_3D_NAMESPACE_END
