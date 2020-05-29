//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "ObstaclesUpdate.h"
#include "../obstacles/ObstacleVector.h"
#include "../utils/MatArrayMath.h"

// define this to update obstacles with old (mrag-like) approach of integrating
// momenta contained in chi before the penalization step:

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

using CHIMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE];
using UDEFMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][3];

template<bool implicitPenalization>
struct KernelIntegrateFluidMomenta : public ObstacleVisitor
{
  const double lambda, dt;
  ObstacleVector * const obstacle_vector;
  const cubism::BlockInfo * info_ptr = nullptr;
  double dvol(const BlockInfo&I, const int x, const int y, const int z) const {
    double h[3]; I.spacing(h, x, y, z);
    return h[0] * h[1] * h[2];
  }

  KernelIntegrateFluidMomenta(double _dt, double _lambda, ObstacleVector* ov)
    : lambda(_lambda), dt(_dt), obstacle_vector(ov) {}

  void operator()(const cubism::BlockInfo& info)
  {
    // first store the lab and info, then do visitor
    assert(info_ptr == nullptr);
    info_ptr = & info;
    ObstacleVisitor* const base = static_cast<ObstacleVisitor*> (this);
    assert( base not_eq nullptr );
    obstacle_vector->Accept( base );
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
    const FluidBlock &b = *(FluidBlock *)info.ptrBlock;
    const CHIMAT & __restrict__ CHI = o->chi;
    double &VV = o->V;
    double &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    double &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    VV = 0; FX = 0; FY = 0; FZ = 0; TX = 0; TY = 0; TZ = 0;
    double &J0 = o->J0, &J1 = o->J1, &J2 = o->J2;
    double &J3 = o->J3, &J4 = o->J4, &J5 = o->J5;
    J0 = 0; J1 = 0; J2 = 0; J3 = 0; J4 = 0; J5 = 0;

    const UDEFMAT & __restrict__ UDEF = o->udef;
    const Real lambdt = lambda*dt;
    if(implicitPenalization)
    {
      o->GfX = 0;
      o->GpX = 0; o->GpY = 0; o->GpZ = 0;
      o->Gj0 = 0; o->Gj1 = 0; o->Gj2 = 0;
      o->Gj3 = 0; o->Gj4 = 0; o->Gj5 = 0;
      o->GuX = 0; o->GuY = 0; o->GuZ = 0;
      o->GaX = 0; o->GaY = 0; o->GaZ = 0;
    }

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

      FX += X * dv * b(ix,iy,iz).u;
      FY += X * dv * b(ix,iy,iz).v;
      FZ += X * dv * b(ix,iy,iz).w;
      TX += X * dv * ( p[1] * b(ix,iy,iz).w - p[2] * b(ix,iy,iz).v );
      TY += X * dv * ( p[2] * b(ix,iy,iz).u - p[0] * b(ix,iy,iz).w );
      TZ += X * dv * ( p[0] * b(ix,iy,iz).v - p[1] * b(ix,iy,iz).u );

      if(implicitPenalization)
      {
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
          b(ix,iy,iz).u - UDEF[iz][iy][ix][0],
          b(ix,iy,iz).v - UDEF[iz][iy][ix][1],
          b(ix,iy,iz).w - UDEF[iz][iy][ix][2]
        };
        o->GuX += penalFac * DiffU[0];
        o->GuY += penalFac * DiffU[1];
        o->GuZ += penalFac * DiffU[2];
        o->GaX += penalFac * ( p[1] * DiffU[2] - p[2] * DiffU[1] );
        o->GaY += penalFac * ( p[2] * DiffU[0] - p[0] * DiffU[2] );
        o->GaZ += penalFac * ( p[0] * DiffU[1] - p[1] * DiffU[0] );
      }
    }
  }
};

template<bool implicitPenalization>
struct KernelFinalizeObstacleVel : public ObstacleVisitor
{
  const double dt, lambda;
  FluidGridMPI * const grid;

  KernelFinalizeObstacleVel(double _dt, double _lambda, FluidGridMPI*g) :
    dt(_dt), lambda(_lambda), grid(g) { }

  void visit(Obstacle* const obst)
  {
    static constexpr int nQoI = 29;
    double M[nQoI] = { 0 };
    const auto& oBlock = obst->getObstacleBlocks();
    #pragma omp parallel for schedule(static,1) reduction(+ : M[:nQoI])
    for (size_t i=0; i<oBlock.size(); i++) {
      if(oBlock[i] == nullptr) continue;
      int k = 0;
      M[k++] += oBlock[i]->V ;
      M[k++] += oBlock[i]->FX; M[k++] += oBlock[i]->FY; M[k++] += oBlock[i]->FZ;
      M[k++] += oBlock[i]->TX; M[k++] += oBlock[i]->TY; M[k++] += oBlock[i]->TZ;
      M[k++] += oBlock[i]->J0; M[k++] += oBlock[i]->J1; M[k++] += oBlock[i]->J2;
      M[k++] += oBlock[i]->J3; M[k++] += oBlock[i]->J4; M[k++] += oBlock[i]->J5;
      if(implicitPenalization) {
      M[k++] +=oBlock[i]->GfX;
      M[k++] +=oBlock[i]->GpX; M[k++] +=oBlock[i]->GpY; M[k++] +=oBlock[i]->GpZ;
      M[k++] +=oBlock[i]->Gj0; M[k++] +=oBlock[i]->Gj1; M[k++] +=oBlock[i]->Gj2;
      M[k++] +=oBlock[i]->Gj3; M[k++] +=oBlock[i]->Gj4; M[k++] +=oBlock[i]->Gj5;
      M[k++] +=oBlock[i]->GuX; M[k++] +=oBlock[i]->GuY; M[k++] +=oBlock[i]->GuZ;
      M[k++] +=oBlock[i]->GaX; M[k++] +=oBlock[i]->GaY; M[k++] +=oBlock[i]->GaZ;
      assert(k==29);
      } else  assert(k==13);
    }
    const auto comm = grid->getCartComm();
    MPI_Allreduce(MPI_IN_PLACE, M, nQoI, MPI_DOUBLE, MPI_SUM, comm);

    #ifndef NDEBUG
      const Real J_magnitude = obst->J[0] + obst->J[1] + obst->J[2];
      static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
    #endif
    assert(std::fabs(obst->mass - M[ 0]) < 10 * EPS * obst->mass);
    assert(std::fabs(obst->J[0] - M[ 7]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[1] - M[ 8]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[2] - M[ 9]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[3] - M[10]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[4] - M[11]) < 10 * EPS * J_magnitude);
    assert(std::fabs(obst->J[5] - M[12]) < 10 * EPS * J_magnitude);
    assert(M[0] > EPS);

    if(implicitPenalization) {
      obst->penalM    = M[13];
      obst->penalCM   = { M[14], M[15], M[16] };
      obst->penalJ    = { M[17], M[18], M[19], M[20], M[21], M[22] };
      obst->penalLmom = { M[23], M[24], M[25] };
      obst->penalAmom = { M[26], M[27], M[28] };
    } else {
      obst->penalM    = M[0];
      obst->penalCM   = { 0, 0, 0 };
      obst->penalJ    = { M[ 7], M[ 8], M[ 9], M[10], M[11], M[12] };
      obst->penalLmom = { M[1], M[2], M[3] };
      obst->penalAmom = { M[4], M[5], M[6] };
    }

    obst->computeVelocities();
  }
};

}  // Anonymous namespace.

void UpdateObstacles::operator()(const double dt)
{
  if(sim.obstacle_vector->nObstacles() == 0) return;

  sim.startProfiler("Obst Int Vel");
  { // integrate momenta by looping over grid
    #pragma omp parallel
    { // each thread needs to call its own non-const operator() function
      //if(0) {
      if(sim.bImplicitPenalization) {
        KernelIntegrateFluidMomenta<1> K(dt, sim.lambda, sim.obstacle_vector);
        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
      } else {
        KernelIntegrateFluidMomenta<0> K(dt, sim.lambda, sim.obstacle_vector);
        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
      }
    }
  }
  sim.stopProfiler();

  sim.startProfiler("Obst Upd Vel");
  //if(0) {
  if(sim.bImplicitPenalization) {
    ObstacleVisitor*K= new KernelFinalizeObstacleVel<1>(dt,sim.lambda,sim.grid);
    sim.obstacle_vector->Accept(K); // accept you son of a french cow
    delete K;
  } else {
    ObstacleVisitor*K= new KernelFinalizeObstacleVel<0>(dt,sim.lambda,sim.grid);
    sim.obstacle_vector->Accept(K); // accept you son of a french cow
    delete K;
  }
  sim.stopProfiler();

  check("UpdateObstacles");
}

CubismUP_3D_NAMESPACE_END
