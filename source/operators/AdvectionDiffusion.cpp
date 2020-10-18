//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "AdvectionDiffusion.h"
#include "../obstacles/ObstacleVector.h"

#include <functional>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

namespace {

enum StepType { Euler = 0, RK1, RK2 };

// input : field with ghosts from which finite differences are computed
template<StepType step, int i> Real& inp(LabMPI& L, const int ix, const int iy, const int iz);
// out   : output field (without ghosts) onto which we save the updated field
template<StepType step, int i> Real& out(FluidBlock& L, const int ix, const int iy, const int iz);
// field : field (with or without ghosts) that we are updating with the operation
template<StepType step, int i> Real& field(LabMPI&, FluidBlock&, const int ix, const int iy, const int iz);
// stencil : which components of the field are read for finite differences
template<StepType step> std::vector<int> stencilFields();

template<StepType step, typename Discretization>
struct KernelAdvectDiffuse : public Discretization
{
  KernelAdvectDiffuse(const SimulationData&s) : Discretization(s), sim(s) {
    //printf("%d %d %e %e %e %e %e %e %e %e\n", loopBeg, loopEnd, CFL,
    //  norUinf, fadeW, fadeS, fadeF, fadeE, fadeN, fadeB);
  }

  const SimulationData & sim;
  const Real dt = sim.dt, mu = sim.nu;
  const std::array<Real, 3>& uInf = sim.uinf;
  const int loopBeg = this->getStencilBeg(), loopEnd = CUP_BLOCK_SIZE-1 + this->getStencilEnd();
  const Real CFL = std::min((Real)1, sim.uMax_measured * sim.dt / sim.hmean);
  const Real norUinf = 1 / std::max({std::fabs(uInf[0]), std::fabs(uInf[1]), std::fabs(uInf[2]), EPS});
  const StencilInfo stencil{this->getStencilBeg(), this->getStencilBeg(), this->getStencilBeg(),
                            this->getStencilEnd(), this->getStencilEnd(), this->getStencilEnd(),
                            false, stencilFields<step>()};

  void applyBCwest(const BlockInfo & I, LabMPI & L) const {
    if (sim.BCx_flag == wall || sim.BCx_flag == periodic) return;
    else if (I.index[0] not_eq 0) return; // not near boundary
    const Real fadeW = 1 - CFL*std::pow(std::max(uInf[0],(Real)0) * norUinf, 2);
    if (fadeW >= 1) return; // no momentum killing at this boundary
    for (int ix = loopBeg; ix < 0; ++ix) {
      const Real fac = std::pow(fadeW, 0 - ix);
      assert(fac <= 1 && fac >= 0);
      for (int iz = loopBeg; iz < loopEnd; ++iz)
      for (int iy = loopBeg; iy < loopEnd; ++iy) {
        inp<step,0>(L,ix,iy,iz) *= fac; inp<step,1>(L,ix,iy,iz) *= fac; inp<step,2>(L,ix,iy,iz) *= fac;
      }
    }
  }

  void applyBCeast(const BlockInfo & I, LabMPI & L) const {
    if (sim.BCx_flag == wall || sim.BCx_flag == periodic) return;
    else if (I.index[0] not_eq sim.bpdx - 1) return; // not near boundary
    const Real fadeE = 1 - CFL*std::pow(std::min(uInf[0],(Real)0) * norUinf, 2);
    if (fadeE >= 1) return; // no momentum killing at this boundary
    for (int ix = CUP_BLOCK_SIZE; ix < loopEnd; ++ix) {
      const Real fac = std::pow(fadeE, ix - CUP_BLOCK_SIZE + 1);
      assert(fac <= 1 && fac >= 0);
      for (int iz = loopBeg; iz < loopEnd; ++iz)
      for (int iy = loopBeg; iy < loopEnd; ++iy) {
        inp<step,0>(L,ix,iy,iz) *= fac; inp<step,1>(L,ix,iy,iz) *= fac; inp<step,2>(L,ix,iy,iz) *= fac;
      }
    }
  }

  void applyBCsouth(const BlockInfo & I, LabMPI & L) const {
    if (sim.BCy_flag == wall || sim.BCy_flag == periodic) return;
    else if (I.index[1] not_eq 0) return; // not near boundary
    const Real fadeS = 1 - CFL*std::pow(std::max(uInf[1],(Real)0) * norUinf, 2);
    if (fadeS >= 1) return; // no momentum killing at this boundary
    for (int iy = loopBeg; iy < 0; ++iy) {
      const Real fac = std::pow(fadeS, 0 - iy);
      assert(fac <= 1 && fac >= 0);
      for (int iz = loopBeg; iz < loopEnd; ++iz)
      for (int ix = loopBeg; ix < loopEnd; ++ix) {
        inp<step,0>(L,ix,iy,iz) *= fac; inp<step,1>(L,ix,iy,iz) *= fac; inp<step,2>(L,ix,iy,iz) *= fac;
      }
    }
  }

  void applyBCnorth(const BlockInfo & I, LabMPI & L) const {
    if (sim.BCy_flag == wall || sim.BCy_flag == periodic) return;
    else if (I.index[1] not_eq sim.bpdy - 1) return; // not near boundary
    const Real fadeN = 1 - CFL*std::pow(std::min(uInf[1],(Real)0) * norUinf, 2);
    if (fadeN >= 1) return; // no momentum killing at this boundary
    for (int iy = CUP_BLOCK_SIZE; iy < loopEnd; ++iy) {
      const Real fac = std::pow(fadeN, iy - CUP_BLOCK_SIZE + 1);
      assert(fac <= 1 && fac >= 0);
      for (int iz = loopBeg; iz < loopEnd; ++iz)
      for (int ix = loopBeg; ix < loopEnd; ++ix) {
        inp<step,0>(L,ix,iy,iz) *= fac; inp<step,1>(L,ix,iy,iz) *= fac; inp<step,2>(L,ix,iy,iz) *= fac;
      }
    }
  }

  void applyBCfront(const BlockInfo & I, LabMPI & L) const {
    if (sim.BCz_flag == wall || sim.BCz_flag == periodic) return;
    else if (I.index[2] not_eq 0) return; // not near boundary
    const Real fadeF = 1 - CFL*std::pow(std::max(uInf[2],(Real)0) * norUinf, 2);
    if (fadeF >= 1) return; // no momentum killing at this boundary
    for (int iz = loopBeg; iz < 0; ++iz) {
      const Real fac = std::pow(fadeF, 0 - iz);
      assert(fac <= 1 && fac >= 0);
      for (int iy = loopBeg; iy < loopEnd; ++iy)
      for (int ix = loopBeg; ix < loopEnd; ++ix) {
        inp<step,0>(L,ix,iy,iz) *= fac; inp<step,1>(L,ix,iy,iz) *= fac; inp<step,2>(L,ix,iy,iz) *= fac;
      }
    }
  }

  void applyBCback(const BlockInfo & I, LabMPI & L) const {
    if (sim.BCz_flag == wall || sim.BCz_flag == periodic) return;
    else if (I.index[2] not_eq sim.bpdz - 1) return; // not near boundary
    const Real fadeB = 1 - CFL*std::pow(std::min(uInf[2],(Real)0) * norUinf, 2);
    if (fadeB >= 1) return; // no momentum killing at this boundary
    for (int iz = CUP_BLOCK_SIZE; iz < loopEnd; ++iz) {
      const Real fac = std::pow(fadeB, iz - CUP_BLOCK_SIZE + 1);
      assert(fac <= 1 && fac >= 0);
      for (int iy = loopBeg; iy < loopEnd; ++iy)
      for (int ix = loopBeg; ix < loopEnd; ++ix) {
        inp<step,0>(L,ix,iy,iz) *= fac; inp<step,1>(L,ix,iy,iz) *= fac; inp<step,2>(L,ix,iy,iz) *= fac;
      }
    }
  }

  void operator()(LabMPI & lab, const BlockInfo& info, FluidBlock& o) const
  {
    const Real facA = this->template advectionCoef<step>(dt, info.h_gridpoint);
    const Real facD = this->template diffusionCoef<step>(dt, info.h_gridpoint, mu);
    applyBCwest(info, lab); applyBCsouth(info, lab); applyBCfront(info, lab);
    applyBCeast(info, lab); applyBCnorth(info, lab); applyBCback(info, lab);
    for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for (int iy=0; iy<FluidBlock::sizeY; ++iy)
    for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
      const Real uAbs[3] = { inp<step,0>(lab,ix,iy,iz) + uInf[0],
                             inp<step,1>(lab,ix,iy,iz) + uInf[1],
                             inp<step,2>(lab,ix,iy,iz) + uInf[2] };
      const Real dudx = this->template diffx<step,0>(lab, o, uAbs, ix, iy, iz);
      const Real dvdx = this->template diffx<step,1>(lab, o, uAbs, ix, iy, iz);
      const Real dwdx = this->template diffx<step,2>(lab, o, uAbs, ix, iy, iz);
      const Real dudy = this->template diffy<step,0>(lab, o, uAbs, ix, iy, iz);
      const Real dvdy = this->template diffy<step,1>(lab, o, uAbs, ix, iy, iz);
      const Real dwdy = this->template diffy<step,2>(lab, o, uAbs, ix, iy, iz);
      const Real dudz = this->template diffz<step,0>(lab, o, uAbs, ix, iy, iz);
      const Real dvdz = this->template diffz<step,1>(lab, o, uAbs, ix, iy, iz);
      const Real dwdz = this->template diffz<step,2>(lab, o, uAbs, ix, iy, iz);
      const Real duD = this->template lap<step,0>(lab, o, ix, iy, iz);
      const Real dvD = this->template lap<step,1>(lab, o, ix, iy, iz);
      const Real dwD = this->template lap<step,2>(lab, o, ix, iy, iz);
      const Real duA = uAbs[0] * dudx + uAbs[1] * dudy + uAbs[2] * dudz;
      const Real dvA = uAbs[0] * dvdx + uAbs[1] * dvdy + uAbs[2] * dvdz;
      const Real dwA = uAbs[0] * dwdx + uAbs[1] * dwdy + uAbs[2] * dwdz;
      out<step,0>(o,ix,iy,iz) = field<step,0>(lab,o,ix,iy,iz) + facA*duA + facD*duD;
      out<step,1>(o,ix,iy,iz) = field<step,1>(lab,o,ix,iy,iz) + facA*dvA + facD*dvD;
      out<step,2>(o,ix,iy,iz) = field<step,2>(lab,o,ix,iy,iz) + facA*dwA + facD*dwD;
    }
  }
};

template<> inline Real& inp<Euler,0>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).u; }
template<> inline Real& inp<Euler,1>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).v; }
template<> inline Real& inp<Euler,2>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).w; }
template<> inline Real& out<Euler,0>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).tmpU; }
template<> inline Real& out<Euler,1>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).tmpV; }
template<> inline Real& out<Euler,2>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).tmpW; }
template<> inline Real& field<Euler,0>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return L(ix,iy,iz).u; }
template<> inline Real& field<Euler,1>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return L(ix,iy,iz).v; }
template<> inline Real& field<Euler,2>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return L(ix,iy,iz).w; }

template<> inline Real& inp<RK1,0>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).u; }
template<> inline Real& inp<RK1,1>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).v; }
template<> inline Real& inp<RK1,2>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).w; }
template<> inline Real& out<RK1,0>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).tmpU; }
template<> inline Real& out<RK1,1>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).tmpV; }
template<> inline Real& out<RK1,2>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).tmpW; }
template<> inline Real& field<RK1,0>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return L(ix,iy,iz).u; }
template<> inline Real& field<RK1,1>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return L(ix,iy,iz).v; }
template<> inline Real& field<RK1,2>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return L(ix,iy,iz).w; }

template<> inline Real& inp<RK2,0>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).tmpU; }
template<> inline Real& inp<RK2,1>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).tmpV; }
template<> inline Real& inp<RK2,2>(LabMPI& L, const int ix, const int iy, const int iz) { return L(ix,iy,iz).tmpW; }
template<> inline Real& out<RK2,0>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).u; }
template<> inline Real& out<RK2,1>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).v; }
template<> inline Real& out<RK2,2>(FluidBlock& o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).w; }
template<> inline Real& field<RK2,0>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).u; }
template<> inline Real& field<RK2,1>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).v; }
template<> inline Real& field<RK2,2>(LabMPI&L, FluidBlock&o, const int ix, const int iy, const int iz) { return o(ix,iy,iz).w; }

template<> inline std::vector<int> stencilFields<Euler>() { return {FE_U, FE_V, FE_W}; }
template<> inline std::vector<int> stencilFields<RK1>() { return {FE_U, FE_V, FE_W}; }
template<> inline std::vector<int> stencilFields<RK2>() { return {FE_TMPU, FE_TMPV, FE_TMPW}; }

struct Central
{
  Central(const SimulationData& s) {}
  template<StepType step, int dir> Real diffx(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    return inp<step,dir>(L,ix+1,iy,iz) - inp<step,dir>(L,ix-1,iy,iz);
  }
  template<StepType step, int dir> Real diffy(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    return inp<step,dir>(L,ix,iy+1,iz) - inp<step,dir>(L,ix,iy-1,iz);
  }
  template<StepType step, int dir> Real diffz(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    return inp<step,dir>(L,ix,iy,iz+1) - inp<step,dir>(L,ix,iy,iz-1);
  }
  template<StepType step, int dir> Real   lap(LabMPI& L, const FluidBlock& o, const int ix, const int iy, const int iz) const {
    return  inp<step,dir>(L,ix+1,iy,iz) + inp<step,dir>(L,ix-1,iy,iz)
          + inp<step,dir>(L,ix,iy+1,iz) + inp<step,dir>(L,ix,iy-1,iz)
          + inp<step,dir>(L,ix,iy,iz+1) + inp<step,dir>(L,ix,iy,iz-1)
          - 6 * inp<step,dir>(L,ix,iy,iz);
  }
  template<StepType step> Real advectionCoef(const Real dt, const Real h) const;
  template<StepType step> Real diffusionCoef(const Real dt, const Real h, const Real mu) const;

  int getStencilBeg() const { return -1; }
  int getStencilEnd() const { return  2; }
};

template<> inline Real Central::advectionCoef<Euler>(const Real dt, const Real h) const {
  return -dt/(2*h);
}
template<> inline Real Central::advectionCoef<RK1>(const Real dt, const Real h) const {
  return -dt/(4*h);
}
template<> inline Real Central::advectionCoef<RK2>(const Real dt, const Real h) const {
  return -dt/(2*h);
}
template<> inline Real Central::diffusionCoef<Euler>(const Real dt, const Real h, const Real mu) const {
  return (mu/h) * (dt/h);
}
template<> inline Real Central::diffusionCoef<RK1>(const Real dt, const Real h, const Real mu) const {
  return (mu/h) * (dt/h) / 2;
}
template<> inline Real Central::diffusionCoef<RK2>(const Real dt, const Real h, const Real mu) const {
  return (mu/h) * (dt/h);
}

struct CentralStretched
{
  CentralStretched(const SimulationData& s) {}
  template<StepType step, int dir> Real diffx(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    const Real um1 = inp<step,dir>(L,ix-1,iy,iz);
    const Real ucc = inp<step,dir>(L,ix,iy,iz);
    const Real up1 = inp<step,dir>(L,ix+1,iy,iz);
    return __FD_2ND(ix, o.fd_cx.first, um1, ucc, up1);
  }
  template<StepType step, int dir> Real diffy(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    const Real um1 = inp<step,dir>(L,ix,iy-1,iz);
    const Real ucc = inp<step,dir>(L,ix,iy,iz);
    const Real up1 = inp<step,dir>(L,ix,iy+1,iz);
    return __FD_2ND(iy, o.fd_cy.first, um1, ucc, up1);
  }
  template<StepType step, int dir> Real diffz(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    const Real um1 = inp<step,dir>(L,ix,iy,iz-1);
    const Real ucc = inp<step,dir>(L,ix,iy,iz);
    const Real up1 = inp<step,dir>(L,ix,iy,iz+1);
    return __FD_2ND(iz, o.fd_cz.first, um1, ucc, up1);
  }
  template<StepType step, int dir> Real   lap(LabMPI& L, const FluidBlock& o, const int ix, const int iy, const int iz) const {
    const Real um1x = inp<step,dir>(L,ix-1,iy,iz), up1x = inp<step,dir>(L,ix+1,iy,iz);
    const Real um1y = inp<step,dir>(L,ix,iy-1,iz), up1y = inp<step,dir>(L,ix,iy+1,iz);
    const Real um1z = inp<step,dir>(L,ix,iy,iz-1), up1z = inp<step,dir>(L,ix,iy,iz+1);
    const Real ucc = inp<step,dir>(L,ix,iy,iz);
    const Real d2dx2 = __FD_2ND(ix, o.fd_cx.second, um1x, ucc, up1x);
    const Real d2dy2 = __FD_2ND(iy, o.fd_cy.second, um1y, ucc, up1y);
    const Real d2dz2 = __FD_2ND(iz, o.fd_cz.second, um1z, ucc, up1z);
    return d2dx2 + d2dy2 + d2dz2;
  }
  template<StepType step> Real advectionCoef(const Real dt, const Real h) const;
  template<StepType step> Real diffusionCoef(const Real dt, const Real h, const Real mu) const;

  int getStencilBeg() const { return -1; }
  int getStencilEnd() const { return  2; }
};

template<> inline Real CentralStretched::advectionCoef<Euler>(const Real dt, const Real h) const {
  return -dt;
}
template<> inline Real CentralStretched::diffusionCoef<Euler>(const Real dt, const Real h, const Real mu) const {
  return dt * mu;
}
/* // unused
template<> inline Real CentralStretched::advectionCoef<RK1>(const Real dt, const Real h) const {
  return -dt / 2;
}
template<> inline Real CentralStretched::advectionCoef<RK2>(const Real dt, const Real h) const {
  return -dt;
}
template<> inline Real CentralStretched::diffusionCoef<RK1>(const Real dt, const Real h, const Real mu) const {
  return dt * mu / 2;
}
template<> inline Real CentralStretched::diffusionCoef<RK2>(const Real dt, const Real h, const Real mu) const {
  return dt * mu;
}
*/

struct Upwind3rd
{
  const SimulationData& _sim;
  const Real invU = 1 / std::max(EPS, _sim.uMax_measured);
  Upwind3rd(const SimulationData& s) : _sim(s) {}

  template<StepType step, int dir>
  inline Real diffx(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    const Real ucc = inp<step,dir>(L,ix,iy,iz);
    const Real um1 = inp<step,dir>(L,ix-1,iy,iz), um2 = inp<step,dir>(L,ix-2,iy,iz);
    const Real up1 = inp<step,dir>(L,ix+1,iy,iz), up2 = inp<step,dir>(L,ix+2,iy,iz);
    #ifndef ADV_3RD_UPWIND
      const Real ddxM = 2*up1 +3*ucc -6*um1 +um2, ddxP = -up2 +6*up1 -3*ucc -2*um1;
      const Real ddxC = up1 - um1, U = std::min((Real)1, std::max(uAbs[0]*invU, (Real)-1));
      const Real UP = std::max((Real)0, U), UM = - std::min((Real)0, U);
      assert(UP>=0 && UP<=1 && UM>=0 && UM<=1 && U>=-1 && U<=1 && std::fabs(UP+UM-std::fabs(U))<EPS);
      //return UP * ddxM + UM * ddxP + (1 - std::fabs(U)) * 3 * ddxC;
      return UP*UP * ddxM + UM*UM * ddxP + (1 - U*U) * 3 * ddxC;
    #else
      return uAbs[0]>0? 2*up1 +3*ucc -6*um1 +um2 : -up2 +6*up1 -3*ucc -2*um1;
    #endif
  }
  template<StepType step, int dir>
  inline Real diffy(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    const Real ucc = inp<step,dir>(L,ix,iy,iz);
    const Real um1 = inp<step,dir>(L,ix,iy-1,iz), um2 = inp<step,dir>(L,ix,iy-2,iz);
    const Real up1 = inp<step,dir>(L,ix,iy+1,iz), up2 = inp<step,dir>(L,ix,iy+2,iz);
    #ifndef ADV_3RD_UPWIND
      const Real ddxM = 2*up1 +3*ucc -6*um1 +um2, ddxP = -up2 +6*up1 -3*ucc -2*um1;
      const Real ddxC = up1 - um1, U = std::min((Real)1, std::max(uAbs[1]*invU, (Real)-1));
      const Real UP = std::max((Real)0, U), UM = - std::min((Real)0, U);
      assert(UP>=0 && UP<=1 && UM>=0 && UM<=1 && U>=-1 && U<=1 && std::fabs(UP+UM-std::fabs(U))<EPS);
      //return UP * ddxM + UM * ddxP + (1 - std::fabs(U)) * 3 * ddxC;
      return UP*UP * ddxM + UM*UM * ddxP + (1 - U*U) * 3 * ddxC;
    #else
      return uAbs[1]>0? 2*up1 +3*ucc -6*um1 +um2 : -up2 +6*up1 -3*ucc -2*um1;
    #endif
  }
  template<StepType step, int dir>
  inline Real diffz(LabMPI& L, const FluidBlock& o, const Real uAbs[3], const int ix, const int iy, const int iz) const {
    const Real ucc = inp<step,dir>(L,ix,iy,iz);
    const Real um1 = inp<step,dir>(L,ix,iy,iz-1), um2 = inp<step,dir>(L,ix,iy,iz-2);
    const Real up1 = inp<step,dir>(L,ix,iy,iz+1), up2 = inp<step,dir>(L,ix,iy,iz+2);
    #ifndef ADV_3RD_UPWIND
      const Real ddxM = 2*up1 +3*ucc -6*um1 +um2, ddxP = -up2 +6*up1 -3*ucc -2*um1;
      const Real ddxC = up1 - um1, U = std::min((Real)1, std::max(uAbs[2]*invU, (Real)-1));
      const Real UP = std::max((Real)0, U), UM = - std::min((Real)0, U);
      assert(UP>=0 && UP<=1 && UM>=0 && UM<=1 && U>=-1 && U<=1 && std::fabs(UP+UM-std::fabs(U))<EPS);
      //return UP * ddxM + UM * ddxP + (1 - std::fabs(U)) * 3 * ddxC;
      return UP*UP * ddxM + UM*UM * ddxP + (1 - U*U) * 3 * ddxC;
    #else
      return uAbs[2]>0? 2*up1 +3*ucc -6*um1 +um2 : -up2 +6*up1 -3*ucc -2*um1;
    #endif
  }
  template<StepType step, int dir>
  inline Real   lap(LabMPI& L, const FluidBlock& o, const int ix, const int iy, const int iz) const {
    return  inp<step,dir>(L,ix+1,iy,iz) + inp<step,dir>(L,ix-1,iy,iz)
          + inp<step,dir>(L,ix,iy+1,iz) + inp<step,dir>(L,ix,iy-1,iz)
          + inp<step,dir>(L,ix,iy,iz+1) + inp<step,dir>(L,ix,iy,iz-1)
          - 6 * inp<step,dir>(L,ix,iy,iz);
  }
  template<StepType step> Real advectionCoef(const Real dt, const Real h) const;
  template<StepType step> Real diffusionCoef(const Real dt, const Real h, const Real mu) const;

  int getStencilBeg() const { return -2; }
  int getStencilEnd() const { return  3; }
};

template<> inline Real Upwind3rd::advectionCoef<Euler>(const Real dt, const Real h) const {
  return -dt/(6*h);
}
template<> inline Real Upwind3rd::advectionCoef<RK1>(const Real dt, const Real h) const {
  return -dt/(12*h);
}
template<> inline Real Upwind3rd::advectionCoef<RK2>(const Real dt, const Real h) const {
  return -dt/(6*h);
}
template<> inline Real Upwind3rd::diffusionCoef<Euler>(const Real dt, const Real h, const Real mu) const {
  return (mu/h) * (dt/h);
}
template<> inline Real Upwind3rd::diffusionCoef<RK1>(const Real dt, const Real h, const Real mu) const {
  return (mu/h) * (dt/h) / 2;
}
template<> inline Real Upwind3rd::diffusionCoef<RK2>(const Real dt, const Real h, const Real mu) const {
  return (mu/h) * (dt/h);
}

struct UpdateAndCorrectInflow
{
  SimulationData & sim;
  FluidGridMPI * const grid = sim.grid;
  const std::vector<cubism::BlockInfo>& vInfo = sim.vInfo();

  static constexpr int BEG = 0, END = CUP_BLOCK_SIZE-1;
  inline bool isW(const BlockInfo&I) const {
    if (sim.BCx_flag == wall || sim.BCx_flag == periodic) return false;
    return I.index[0] == 0;
  };
  inline bool isE(const BlockInfo&I) const {
    if (sim.BCx_flag == wall || sim.BCx_flag == periodic) return false;
    return I.index[0] == sim.bpdx-1;
  };
  inline bool isS(const BlockInfo&I) const {
    if (sim.BCy_flag == wall || sim.BCy_flag == periodic) return false;
    return I.index[1] == 0;
  };
  inline bool isN(const BlockInfo&I) const {
    if (sim.BCy_flag == wall || sim.BCy_flag == periodic) return false;
    return I.index[1] == sim.bpdy-1;
  };
  inline bool isF(const BlockInfo&I) const {
    if (sim.BCz_flag == wall || sim.BCz_flag == periodic) return false;
    return I.index[2] == 0;
  };
  inline bool isB(const BlockInfo&I) const {
    if (sim.BCz_flag == wall || sim.BCz_flag == periodic) return false;
    return I.index[2] == sim.bpdz-1;
  };

  UpdateAndCorrectInflow(SimulationData & s) : sim(s) { }

  template<bool transferTmp, bool nonuniform = false> void operate() const
  {
    double sumInflow = 0, throughFlow = 0;
    #pragma omp parallel for schedule(static) reduction(+:sumInflow,throughFlow)
    for(size_t i=0; i<vInfo.size(); i++)
    {
      FluidBlock& b = *(FluidBlock*) vInfo[i].ptrBlock;
      if(transferTmp)
        for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for(int iy=0; iy<FluidBlock::sizeY; ++iy)
        for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
          b(ix,iy,iz).u = b(ix,iy,iz).tmpU;
          b(ix,iy,iz).v = b(ix,iy,iz).tmpV;
          b(ix,iy,iz).w = b(ix,iy,iz).tmpW;
        }

      if(isW(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
                        for (int iy=0; iy<FluidBlock::sizeY; ++iy) {
        sumInflow -= b(BEG,iy,iz).u; throughFlow += std::fabs(b(BEG,iy,iz).u);
      }

      if(isE(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
                        for (int iy=0; iy<FluidBlock::sizeY; ++iy) {
        sumInflow += b(END,iy,iz).u; throughFlow += std::fabs(b(END,iy,iz).u);
      }

      if(isS(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
                        for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        sumInflow -= b(ix,BEG,iz).v; throughFlow += std::fabs(b(ix,BEG,iz).v);
      }

      if(isN(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
                        for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        sumInflow += b(ix,END,iz).v; throughFlow += std::fabs(b(ix,END,iz).v);
      }

      if(isF(vInfo[i])) for (int iy=0; iy<FluidBlock::sizeY; ++iy)
                        for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        sumInflow -= b(ix,iy,BEG).w; throughFlow += std::fabs(b(ix,iy,BEG).w);
      }

      if(isB(vInfo[i])) for (int iy=0; iy<FluidBlock::sizeY; ++iy)
                        for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        sumInflow += b(ix,iy,END).w; throughFlow += std::fabs(b(ix,iy,END).w);
      }
    }

    double sums[2] = {sumInflow, throughFlow};
    MPI_Allreduce(MPI_IN_PLACE, sums,2,MPI_DOUBLE,MPI_SUM, grid->getCartComm());
    const auto nTotX = FluidBlock::sizeX * sim.bpdx;
    const auto nTotY = FluidBlock::sizeY * sim.bpdy;
    const auto nTotZ = FluidBlock::sizeZ * sim.bpdz;

    const Real corr = nonuniform ? sums[0] / std::max((double)EPS, sums[1])
                      : sums[0] / (2*(nTotX*nTotY + nTotX*nTotZ + nTotY*nTotZ));

    const std::function<void(Real&, const Real)> update1 =
      [&] (Real& vel, const Real fac) { vel += fac; };
    const std::function<void(Real&, const Real)> update2 =
      [&] (Real& vel, const Real fac) { vel += fac * std::fabs(vel); };
    const auto update = nonuniform ? update2 : update1;

    if(std::fabs(corr) < EPS) return;
    if(sim.verbose) printf("Inflow correction %e\n", corr);

    #pragma omp parallel for schedule(static)
    for(size_t i=0; i<vInfo.size(); i++) {
      FluidBlock& b = *(FluidBlock*) vInfo[i].ptrBlock;
      if(isW(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for (int iy=0; iy<FluidBlock::sizeY; ++iy) update(b(BEG,iy,iz).u, corr);

      if(isE(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for (int iy=0; iy<FluidBlock::sizeY; ++iy) update(b(END,iy,iz).u,-corr);

      if(isS(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for (int ix=0; ix<FluidBlock::sizeX; ++ix) update(b(ix,BEG,iz).v, corr);

      if(isN(vInfo[i])) for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
        for (int ix=0; ix<FluidBlock::sizeX; ++ix) update(b(ix,END,iz).v,-corr);

      if(isF(vInfo[i])) for (int iy=0; iy<FluidBlock::sizeY; ++iy)
        for (int ix=0; ix<FluidBlock::sizeX; ++ix) update(b(ix,iy,BEG).w, corr);

      if(isB(vInfo[i])) for (int iy=0; iy<FluidBlock::sizeY; ++iy)
        for (int ix=0; ix<FluidBlock::sizeX; ++ix) update(b(ix,iy,END).w,-corr);
    }
  }
};

}

void AdvectionDiffusion::operator()(const double dt)
{
  if(sim.bUseStretchedGrid)
  {
    sim.startProfiler("AdvDiff Kernel");
    const KernelAdvectDiffuse<Euler, CentralStretched> K(sim);
    compute(K);
    sim.stopProfiler();
    sim.startProfiler("AdvDiff copy");
    const UpdateAndCorrectInflow U(sim);
    U.operate<true, true>();
    sim.stopProfiler();
  }
  else
  {
    if(sim.bRungeKutta23) {
      sim.startProfiler("AdvDiff23 Kernel");
      if(sim.bAdvection3rdOrder) {
        const KernelAdvectDiffuse<RK1, Upwind3rd> K1(sim);
        compute(K1);
        const KernelAdvectDiffuse<RK2, Upwind3rd> K2(sim);
        compute(K2);
      } else {
        const KernelAdvectDiffuse<RK1, Central> K1(sim);
        compute(K1);
        const KernelAdvectDiffuse<RK2, Central> K2(sim);
        compute(K2);
      }
      sim.stopProfiler();
      if(not sim.bUseFourierBC) {
        sim.startProfiler("AdvDiff copy");
        const UpdateAndCorrectInflow U(sim);
        U.operate<false>();
        sim.stopProfiler();
      }
    } else {
      sim.startProfiler("AdvDiff Kernel");
      if(sim.bAdvection3rdOrder) {
        const KernelAdvectDiffuse<Euler, Upwind3rd> K(sim);
        compute(K);
      } else {
        const KernelAdvectDiffuse<Euler, Central> K(sim);
        compute(K);
      }
      sim.stopProfiler();
      sim.startProfiler("AdvDiff copy");
      const UpdateAndCorrectInflow U(sim);
      U.operate<true>();
      sim.stopProfiler();
    }
  }
  check("AdvectionDiffusion");
}

CubismUP_3D_NAMESPACE_END
