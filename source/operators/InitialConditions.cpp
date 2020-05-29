//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "InitialConditions.h"
#include "../spectralOperators/SpectralIcGenerator.h"
#include "../obstacles/ObstacleVector.h"

#include <random>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

class KernelIC
{
 public:
  KernelIC(const Real u) {}
  void operator()(const BlockInfo& info, FluidBlock& block) const
  {
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) block(ix,iy,iz).clear();
  }
};

class KernelIC_taylorGreen
{
  const std::array<Real, 3> ext;
  const Real uMax;
  const Real a = 2*M_PI / ext[0], b = 2*M_PI / ext[1], c = 2*M_PI / ext[2];
  const Real A = uMax, B = - uMax * ext[1] / ext[0];
 public:
  KernelIC_taylorGreen(const std::array<Real, 3> &extent, const Real U): ext{extent}, uMax(U) {}
  void operator()(const BlockInfo& info, FluidBlock& block) const
  {
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      block(ix,iy,iz).clear();
      Real p[3]; info.pos(p, ix, iy, iz);
      block(ix,iy,iz).u = A*std::cos(a*p[0])*std::sin(b*p[1])*std::sin(c*p[2]);
      block(ix,iy,iz).v = B*std::sin(a*p[0])*std::cos(b*p[1])*std::sin(c*p[2]);
    }
  }
};

class KernelIC_channel
{
  const int dir;
  const std::array<Real, 3> ext;
  const Real uMax, H = ext[dir], FAC = 4*uMax/H/H; // FAC = 0.5*G/mu
  //umax =  0.5*G/mu * 0.25*H*H
 public:
  KernelIC_channel(const std::array<Real, 3> &extent, const Real U, const int _dir):
    dir(_dir), ext{extent}, uMax(U) {}

  void operator()(const BlockInfo& info, FluidBlock& block) const
  {
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      block(ix,iy,iz).clear();
      Real p[3]; info.pos(p, ix, iy, iz);
      block(ix,iy,iz).u = FAC * p[dir] * (H-p[dir]);
    }
  }
};

class KernelIC_channelrandom
{
  const int dir;
  const std::array<Real, 3> ext;
  const Real uMax, H = ext[dir], FAC = 4*uMax/H/H; // FAC = 0.5*G/mu
  const Real delta_tau = 5.0/180;
  //umax =  0.5*G/mu * 0.25*H*H
 public:
  KernelIC_channelrandom(const std::array<Real, 3> &extent, const Real U, const int _dir):
    dir(_dir), ext{extent}, uMax(U) {}

  void operator()(const BlockInfo& info, FluidBlock& block) const
  {
    std::random_device seed;
    std::mt19937 gen(seed());
    std::normal_distribution<Real> dist(0.0, 0.01);
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      block(ix,iy,iz).clear();
      Real p[3]; info.pos(p, ix, iy, iz);
      const Real U = FAC * p[dir] * (H-p[dir]);
      block(ix,iy,iz).u = U * (1 + dist(gen));
    }
  }
};

struct InitialPenalization : public ObstacleVisitor
{
  FluidGridMPI * const grid;
  const double dt;
  const Real * const uInf;
  const std::vector<BlockInfo>& vInfo = grid->getBlocksInfo();

  InitialPenalization(FluidGridMPI*g, const double _dt,
    const Real*const u) : grid(g), dt(_dt), uInf(u) { }

  void visit(Obstacle* const obstacle)
  {
    using CHI_MAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE];
    using UDEFMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][3];
    #pragma omp parallel
    {
      const auto& obstblocks = obstacle->getObstacleBlocks();
      const std::array<double,3> centerOfMass = obstacle->getCenterOfMass();
      const std::array<double,3> uBody = obstacle->getTranslationVelocity();
      const std::array<double,3> omegaBody = obstacle->getAngularVelocity();

      #pragma omp for schedule(dynamic)
      for (size_t i = 0; i < vInfo.size(); ++i)
      {
        const BlockInfo& info = vInfo[i];
        const auto pos = obstblocks[info.blockID];
        if(pos == nullptr) continue;

        FluidBlock& b = *(FluidBlock*)info.ptrBlock;
        CHI_MAT & __restrict__ CHI = pos->chi;
        UDEFMAT & __restrict__ UDEF = pos->udef;

        for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix)
        {
          Real p[3]; info.pos(p, ix, iy, iz);
          p[0]-=centerOfMass[0]; p[1]-=centerOfMass[1]; p[2]-=centerOfMass[2];
          const Real object_UR[3] = {
              (Real) omegaBody[1]*p[2] - (Real) omegaBody[2]*p[1],
              (Real) omegaBody[2]*p[0] - (Real) omegaBody[0]*p[2],
              (Real) omegaBody[0]*p[1] - (Real) omegaBody[1]*p[0]
          };
          const Real U_TOT[3] = {
              (Real)uBody[0] + object_UR[0] + UDEF[iz][iy][ix][0],
              (Real)uBody[1] + object_UR[1] + UDEF[iz][iy][ix][1],
              (Real)uBody[2] + object_UR[2] + UDEF[iz][iy][ix][2]
          };
          // what if multiple obstacles share a block??
          // let's plus equal and wake up during the night to stress about it
          b(ix,iy,iz).u += CHI[iz][iy][ix] * ( U_TOT[0] - b(ix,iy,iz).u );
          b(ix,iy,iz).v += CHI[iz][iy][ix] * ( U_TOT[1] - b(ix,iy,iz).v );
          b(ix,iy,iz).w += CHI[iz][iy][ix] * ( U_TOT[2] - b(ix,iy,iz).w );
        }
      }
    }
  }
};

}

void InitialConditions::operator()(const double dt)
{
  if(sim.initCond == "zero") {
    if(sim.verbose) printf("Zero-values initial conditions.\n");
    run(KernelIC(0));
  }
  if(sim.initCond == "taylorGreen") {
    if(sim.verbose) printf("Taylor Green vortex initial conditions.\n");
    run(KernelIC_taylorGreen(sim.extent, sim.uMax_forced));
  }
  if(sim.initCond == "channelRandom")
  {
    if(sim.verbose) printf("Channel flow random initial conditions.\n");
    if( sim.BCx_flag == wall ) {
      printf("ERROR: channel flow must be periodic or dirichlet in x.\n");
      fflush(0); abort();
    }
    const bool channelY = sim.BCy_flag==wall, channelZ = sim.BCz_flag==wall;
    if( (channelY && channelZ) or (!channelY && !channelZ) ) {
      printf("ERROR: wrong channel flow BC in y or z.\n");
      fflush(0); abort();
    }
    const int dir = channelY? 1 : 2;
    run(KernelIC_channelrandom(sim.extent, sim.uMax_forced, dir));
  }
  if(sim.initCond == "channel")
  {
    if(sim.verbose) printf("Channel flow initial conditions.\n");
    if( sim.BCx_flag == wall ) {
      printf("ERROR: channel flow must be periodic or dirichlet in x.\n");
      fflush(0); abort();
    }
    const bool channelY = sim.BCy_flag==wall, channelZ = sim.BCz_flag==wall;
    if( (channelY && channelZ) or (!channelY && !channelZ) ) {
      printf("ERROR: wrong channel flow BC in y or z.\n");
      fflush(0); abort();
    }
    const int dir = channelY? 1 : 2;
    run(KernelIC_channel(sim.extent, sim.uMax_forced, dir));
  }
  if (sim.initCond == "HITurbulence")
  {
    if(sim.verbose) printf("Homogeneous Isotropic Turbulence initial conditions.\n");

    SpectralIcGenerator sIc(sim);
    sIc.run();
  }
  {
    //zero fields, going to contain Udef:
    #pragma omp parallel for schedule(static)
    for(unsigned i=0; i<vInfo.size(); i++)
    {
      FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy,iz).tmpU = 0; b(ix,iy,iz).tmpV = 0; b(ix,iy,iz).tmpW = 0;
      }
    }
    //store deformation velocities onto tmp fields:
    ObstacleVisitor* visitor = new InitialPenalization(grid, dt, sim.uinf.data());
    sim.obstacle_vector->Accept(visitor);
    delete visitor;
  }

  check("InitialConditions");
}

CubismUP_3D_NAMESPACE_END
