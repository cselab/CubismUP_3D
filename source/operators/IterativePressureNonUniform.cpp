//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "IterativePressureNonUniform.h"
#ifdef _ACCFFT_
#include "../poisson/PoissonSolverACCPeriodic.h"
#include "../poisson/PoissonSolverACCUnbounded.h"
#else
#include "../poisson/PoissonSolverPeriodic.h"
#include "../poisson/PoissonSolverUnbounded.h"
#endif
// TODO : Cosine transform on GPU!?
#include "../poisson/PoissonSolverMixed.h"


CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

using CHIMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE];
using UDEFMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][3];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
//static constexpr Real DBLEPS = std::numeric_limits<double>::epsilon();

namespace {

class KernelPressureRHS_nonUniform
{
 private:
  const Real invdt, meanh;
  const Real fadeLen[3], ext[3], iFade[3];
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  PoissonSolver * const solver;

  inline bool _is_touching(const FluidBlock& b) const {
    const bool touchW = fadeLen[0] >= b.min_pos[0];
    const bool touchE = fadeLen[0] >= ext[0] - b.max_pos[0];
    const bool touchS = fadeLen[1] >= b.min_pos[1];
    const bool touchN = fadeLen[1] >= ext[1] - b.max_pos[1];
    const bool touchB = fadeLen[2] >= b.min_pos[2];
    const bool touchF = fadeLen[2] >= ext[2] - b.max_pos[2];
    return touchN || touchE || touchS || touchW || touchF || touchB;
  }

  inline Real fade(const BlockInfo&i, const int x,const int y,const int z) const
  {
    Real p[3]; i.pos(p, x, y, z);
    const Real zt = iFade[2] * std::max(Real(0), fadeLen[2] -(ext[2]-p[2]) );
    const Real zb = iFade[2] * std::max(Real(0), fadeLen[2] - p[2] );
    const Real yt = iFade[1] * std::max(Real(0), fadeLen[1] -(ext[1]-p[1]) );
    const Real yb = iFade[1] * std::max(Real(0), fadeLen[1] - p[1] );
    const Real xt = iFade[0] * std::max(Real(0), fadeLen[0] -(ext[0]-p[0]) );
    const Real xb = iFade[0] * std::max(Real(0), fadeLen[0] - p[0] );
    return 1-std::pow(std::min( std::max({zt,zb,yt,yb,xt,xb}), (Real)1), 2);
  }

  inline Real RHSV(Lab&l, const int ix, const int iy, const int iz,
    const BlkCoeffX &cx, const BlkCoeffX &cy, const BlkCoeffX &cz) const
  {
    const FluidElement& L  = l(ix,  iy,  iz);
    const FluidElement& LW = l(ix-1,iy,  iz  ), & LE = l(ix+1,iy,  iz  );
    const FluidElement& LS = l(ix,  iy-1,iz  ), & LN = l(ix,  iy+1,iz  );
    const FluidElement& LF = l(ix,  iy,  iz-1), & LB = l(ix,  iy,  iz+1);
    const Real dudx = __FD_2ND(ix, cx, LW.u, L.u, LE.u);
    const Real dvdy = __FD_2ND(iy, cy, LS.v, L.v, LN.v);
    const Real dwdz = __FD_2ND(iz, cz, LF.w, L.w, LB.w);
    return dudx + dvdy + dwdz;
  }

  inline Real RHSP(Lab&l, const int ix,const int iy,const int iz, const Real h,
   const Real V, const BlkCoeffX&cx,const BlkCoeffX&cy,const BlkCoeffX&cz) const
  {
    const FluidElement& L  = l(ix,  iy,  iz);
    const FluidElement& LW = l(ix-1,iy,  iz  ), & LE = l(ix+1,iy,  iz  );
    const FluidElement& LS = l(ix,  iy-1,iz  ), & LN = l(ix,  iy+1,iz  );
    const FluidElement& LF = l(ix,  iy,  iz-1), & LB = l(ix,  iy,  iz+1);
    assert(cx.c00[ix]<=0 && cy.c00[iy]<=0 && cz.c00[iz]<=0);
    const Real f000 = L.p * (6*h + V * (cx.c00[ix] + cy.c00[iy] + cz.c00[iz]) );
    const Real fm00 = LW.p * (h - V * cx.cm1[ix]);
    const Real fp00 = LE.p * (h - V * cx.cp1[ix]);
    const Real f0m0 = LS.p * (h - V * cy.cm1[iy]);
    const Real f0p0 = LN.p * (h - V * cy.cp1[iy]);
    const Real f00m = LF.p * (h - V * cz.cm1[iz]);
    const Real f00p = LB.p * (h - V * cz.cp1[iz]);
    return fm00 + fp00 + f0m0 + f0p0 + f00m + f00p - f000;
  }

 public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1,2,2,2,false, {FE_U,FE_V,FE_W}};

  KernelPressureRHS_nonUniform(double _dt, double _h, const Real buf[3],
  const std::array<Real, 3> &E, PoissonSolver* ps): invdt(1/_dt),
  meanh(_h), fadeLen{buf[0],buf[1],buf[2]}, ext{E[0],E[1],E[2]},
  iFade{1/(buf[0]+EPS), 1/(buf[1]+EPS), 1/(buf[2]+EPS)}, solver(ps) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o) const
  {
    // FD coefficients for first derivative
    const Real vHat = std::pow(meanh, 3);
    const auto &c1x =o.fd_cx.first,  &c1y =o.fd_cy.first,  &c1z =o.fd_cz.first;
    const auto &c2x =o.fd_cx.second, &c2y =o.fd_cy.second, &c2z =o.fd_cz.second;
    Real* __restrict__ const ret = solver->data + solver->_offset_ext(info);
    const unsigned SX=solver->stridex, SY=solver->stridey, SZ=solver->stridez;
    if( not _is_touching(o) )
    {
      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real RHSV_ = vHat * RHSV(lab, ix,iy,iz, c1x,c1y,c1z) * invdt;
        const Real RHSP_ = RHSP(lab, ix,iy,iz, meanh, vHat, c2x,c2y,c2z);
        ret[SZ*iz + SY*iy + SX*ix] = RHSV_ + RHSP_;
      }
    }
    else
    {
      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        const Real RHSV_ = vHat * RHSV(lab, ix,iy,iz, c1x,c1y,c1z) * invdt;
        const Real RHSP_ = RHSP(lab, ix,iy,iz, meanh, vHat, c2x,c2y,c2z);
        ret[SZ*iz + SY*iy + SX*ix] = fade(info, ix,iy,iz) * (RHSV_ + RHSP_);
      }
    }
  }
};

class KernelGradP_nonUniform
{
  const Real dt;
 public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_P}}};

  KernelGradP_nonUniform(double _dt): dt(_dt) {}

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

IterativePressureNonUniform::IterativePressureNonUniform(SimulationData & s) : Operator(s)
{
  if(sim.bUseFourierBC)
  pressureSolver = new PoissonSolverPeriodic(sim);
  else if (sim.bUseUnboundedBC)
  pressureSolver = new PoissonSolverUnbounded(sim);
  else
  pressureSolver = new PoissonSolverMixed(sim);
  sim.pressureSolver = pressureSolver;
}

void IterativePressureNonUniform::operator()(const double dt)
{
  int iter=0;
  Real relDF = 1e3;
  const Real unifH = sim.uniformH();
  for(iter = 0; iter < 1000; iter++)
  {
    {
      sim.startProfiler("PresRHS Kernel");
      sim.pressureSolver->reset();
      //place onto p: ( div u^(t+1) - div u^* ) / dt
      //where i want div u^(t+1) to be equal to div udef
      const KernelPressureRHS_nonUniform K(dt, unifH, sim.fadeOutLengthPRHS, sim.extent, sim.pressureSolver);
      compute<KernelPressureRHS_nonUniform>(K);
      sim.stopProfiler();
    }

    pressureSolver->solve();

    sim.startProfiler("sol2cub");
    {
      Real err = 0, norm = 0;
      #pragma omp parallel for schedule(static) reduction(+ : err, norm)
      for(size_t i=0; i<vInfo.size(); i++)
      {
        assert((size_t) vInfo[i].blockID == i);
        FluidBlock& b = *(FluidBlock*) vInfo[i].ptrBlock;
        const size_t offset = pressureSolver->_offset_ext(vInfo[i]);
        Real* const ret = pressureSolver->data + offset;
        const unsigned SX = pressureSolver->stridex;
        const unsigned SY = pressureSolver->stridey;
        const unsigned SZ = pressureSolver->stridez;
        for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          b(ix,iy,iz).tmpU = b(ix,iy,iz).p; // stores old pressure
          b(ix,iy,iz).p = ret[SZ*iz + SY*iy + SX*ix]; // stores new pressure
          err += std::pow( b(ix,iy,iz).tmpU - b(ix,iy,iz).p, 2 );
          norm += std::pow( b(ix,iy,iz).p, 2 );
        }
      }
      double M[2] = {(double) err, (double) norm};
      MPI_Allreduce(MPI_IN_PLACE, M, 2, MPI_DOUBLE, MPI_SUM, sim.app_comm);
      relDF = std::sqrt( M[0] / (EPS + M[1]) );
    }
    sim.stopProfiler();

    if(sim.verbose) printf("iter:%02d - max relative error: %f\n", iter, relDF);
    if(iter && relDF < 0.001) break; // do at least 2 iterations
  }

  sim.startProfiler("GradP"); //pressure correction dudt* = - grad P / rho
  {
    const KernelGradP_nonUniform K(dt);
    compute<KernelGradP_nonUniform>(K);
  }
  sim.stopProfiler();


  check("IterativePressureNonUniform");
}

CubismUP_3D_NAMESPACE_END
