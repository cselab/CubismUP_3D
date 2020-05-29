//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "FadeOut.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

class KernelFadeOut
{
 private:
  const Real ext[3], fadeLen[3], iFade[3];
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  bool _is_touching(const FluidBlock& b) const
  {
    const bool touchW = fadeLen[0] >= b.min_pos[0];
    const bool touchE = fadeLen[0] >= ext[0] - b.max_pos[0];
    const bool touchS = fadeLen[1] >= b.min_pos[1];
    const bool touchN = fadeLen[1] >= ext[1] - b.max_pos[1];
    const bool touchB = fadeLen[2] >= b.min_pos[2];
    const bool touchF = fadeLen[2] >= ext[2] - b.max_pos[2];
    return touchN || touchE || touchS || touchW || touchF || touchB;
  }
  Real fade(const BlockInfo&i, const int x, const int y, const int z) const
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

 public:
  KernelFadeOut(const Real buf[3], const Real extent[3]) :
  ext{extent[0],extent[1],extent[2]}, fadeLen{buf[0],buf[1],buf[2]},
  iFade{1/(buf[0]+EPS), 1/(buf[1]+EPS), 1/(buf[2]+EPS)} {}

  void operator()(const BlockInfo& info, FluidBlock& b) const
  {
    if( _is_touching(b) )
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const Real FADE = fade(info, ix, iy, iz);
      b(ix,iy,iz).u *= FADE;
      b(ix,iy,iz).v *= FADE;
      b(ix,iy,iz).w *= FADE;
    }
  }
};

}

void FadeOut::operator()(const double dt)
{
  sim.startProfiler("FadeOut Kernel");
  #pragma omp parallel
  {
    KernelFadeOut kernel(sim.fadeOutLengthU, sim.extent.data());
    #pragma omp for schedule(static)
    for (size_t i=0; i<vInfo.size(); i++)
      kernel(vInfo[i], *(FluidBlock*) vInfo[i].ptrBlock);
  }
  sim.stopProfiler();
  check("FadeOut");
}

void InflowBC::operator()(const double dt)
{
  static constexpr Real EPS = std::numeric_limits<Real>::epsilon();
  static constexpr int BX = 0, EX = FluidBlock::sizeX-1;
  static constexpr int BY = 0, EY = FluidBlock::sizeY-1;
  static constexpr int BZ = 0, EZ = FluidBlock::sizeZ-1;
  const bool applyX = sim.BCx_flag == dirichlet || sim.BCx_flag == freespace;
  const bool applyY = sim.BCy_flag == dirichlet || sim.BCy_flag == freespace;
  const bool applyZ = sim.BCz_flag == dirichlet || sim.BCz_flag == freespace;
  const auto touchW =[&](const BlockInfo&I) {
                                   return applyX && I.index[0]==0;          };
  const auto touchE =[&](const BlockInfo&I) {
                                   return applyX && I.index[0]==sim.bpdx-1; };
  const auto touchS =[&](const BlockInfo&I) {
                                   return applyY && I.index[1]==0;          };
  const auto touchN =[&](const BlockInfo&I) {
                                   return applyY && I.index[1]==sim.bpdy-1; };
  const auto touchB =[&](const BlockInfo&I) {
                                   return applyZ && I.index[2]==0;          };
  const auto touchF =[&](const BlockInfo&I) {
                                   return applyZ && I.index[2]==sim.bpdz-1; };
  const Real UX = sim.uinf[0], UY = sim.uinf[1], UZ = sim.uinf[2];
  const Real norm = std::max( std::sqrt(UX*UX + UY*UY + UZ*UZ), EPS );
  const Real CW = std::max( UX,(Real)0)/norm, CE = std::max(-UX,(Real)0)/norm;
  const Real CS = std::max( UY,(Real)0)/norm, CN = std::max(-UY,(Real)0)/norm;
  const Real CB = std::max( UZ,(Real)0)/norm, CF = std::max(-UZ,(Real)0)/norm;
  sim.startProfiler("FadeOut Kernel");

  #pragma omp parallel for schedule(dynamic)
  for (size_t i=0; i < vInfo.size(); ++i)
  {
    FluidBlock& b = *(FluidBlock*) vInfo[i].ptrBlock;
    if( touchW(vInfo[i]) ) // west
      for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for (int iy=0; iy<FluidBlock::sizeY; ++iy) {
        b(BX,iy,iz).u -= CW * b(BX,iy,iz).u;
        b(BX,iy,iz).v -= CW * b(BX,iy,iz).v;
        b(BX,iy,iz).w -= CW * b(BX,iy,iz).w;
      }
    if( touchE(vInfo[i]) ) // east
      for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for (int iy=0; iy<FluidBlock::sizeY; ++iy) {
        b(EX,iy,iz).u -= CE * b(EX,iy,iz).u;
        b(EX,iy,iz).v -= CE * b(EX,iy,iz).v;
        b(EX,iy,iz).w -= CE * b(EX,iy,iz).w;
      }
    if( touchS(vInfo[i]) ) // south
      for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,BY,iz).u -= CS * b(ix,BY,iz).u;
        b(ix,BY,iz).v -= CS * b(ix,BY,iz).v;
        b(ix,BY,iz).w -= CS * b(ix,BY,iz).w;
      }
    if( touchN(vInfo[i]) ) // north
      for (int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,EY,iz).u -= CN * b(ix,EY,iz).u;
        b(ix,EY,iz).v -= CN * b(ix,EY,iz).v;
        b(ix,EY,iz).w -= CN * b(ix,EY,iz).w;
      }
    if( touchB(vInfo[i]) ) // back
      for (int iy=0; iy<FluidBlock::sizeY; ++iy)
      for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy,BZ).u -= CB * b(ix,iy,BZ).u;
        b(ix,iy,BZ).v -= CB * b(ix,iy,BZ).v;
        b(ix,iy,BZ).w -= CB * b(ix,iy,BZ).w;
      }
    if( touchF(vInfo[i]) ) // front
      for (int iy=0; iy<FluidBlock::sizeY; ++iy)
      for (int ix=0; ix<FluidBlock::sizeX; ++ix) {
        b(ix,iy,EZ).u -= CF * b(ix,iy,EZ).u;
        b(ix,iy,EZ).v -= CF * b(ix,iy,EZ).v;
        b(ix,iy,EZ).w -= CF * b(ix,iy,EZ).w;
      }
  }
  sim.stopProfiler();
  check("FadeOut");
}

CubismUP_3D_NAMESPACE_END
