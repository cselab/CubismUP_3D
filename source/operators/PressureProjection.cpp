//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PressureProjection.h"
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

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

class KernelGradP
{
  const Real dt;
 public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_P}}};

  KernelGradP(double _dt, const std::array<Real, 3> &ext): dt(_dt) {}

  ~KernelGradP() {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    const Real fac = - 0.5 * dt / info.h_gridpoint;
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
       // p contains the pressure correction after the Poisson solver
       o(ix,iy,iz).u += fac*(lab(ix+1,iy,iz).p-lab(ix-1,iy,iz).p);
       o(ix,iy,iz).v += fac*(lab(ix,iy+1,iz).p-lab(ix,iy-1,iz).p);
       o(ix,iy,iz).w += fac*(lab(ix,iy,iz+1).p-lab(ix,iy,iz-1).p);
    }
  }
};

class KernelGradP_nonUniform
{
  const Real dt;
 public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_P}}};

  KernelGradP_nonUniform(double _dt, const std::array<Real, 3> &ext): dt(_dt) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    // FD coefficients for first derivative
    const BlkCoeffX& cx = o.fd_cx.first;
    const BlkCoeffY& cy = o.fd_cy.first;
    const BlkCoeffZ& cz = o.fd_cz.first;
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const FluidElement &L =lab(ix,iy,iz);
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);
     // p contains the pressure correction after the Poisson solver
     o(ix,iy,iz).u -= dt * __FD_2ND(ix, cx, LW.p, L.p, LE.p);
     o(ix,iy,iz).v -= dt * __FD_2ND(iy, cy, LS.p, L.p, LN.p);
     o(ix,iy,iz).w -= dt * __FD_2ND(iz, cz, LF.p, L.p, LB.p);
    }
  }
};

}

PressureProjection::PressureProjection(SimulationData & s) : Operator(s)
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
  else if (sim.useSolver == "petsc") {
    printf("PoissonSolverMixed_PETSC\n"); fflush(0);
    pressureSolver = new PoissonSolverMixed_PETSC(sim);
  }
  #endif
  else
    pressureSolver = new PoissonSolverMixed(sim);
    //pressureSolver = new PoissonSolverPeriodic(sim);
  sim.pressureSolver = pressureSolver;
}

void PressureProjection::operator()(const double dt)
{
  pressureSolver->solve();

  sim.startProfiler("sol2cub");
  pressureSolver->_fftw2cub();
  sim.stopProfiler();

  sim.startProfiler("GradP"); //pressure correction dudt* = - grad P / rho
  if(sim.bUseStretchedGrid)
  {
    const KernelGradP_nonUniform K(dt, sim.extent);
    compute<KernelGradP_nonUniform>(K);
  }
  else
  {
    const KernelGradP K(dt, sim.extent);
    compute<KernelGradP>(K);
  }
  sim.stopProfiler();

  check("PressureProjection");
}

CubismUP_3D_NAMESPACE_END
