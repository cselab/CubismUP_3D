//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "SGS.h"

CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;
//#define DSM_LILLY
//#define DSM_LOCAL

static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

struct SGSHelperElement
{
  typedef Real RealType;
  // Derivatives of nu_sgs
  Real nu=0, duD=0, dvD=0, dwD=0;
  void clear() { nu=0; duD=0; dvD=0; dwD=0; }
  SGSHelperElement(const SGSHelperElement& c) = delete;
};

using SGSBlock   = BaseBlock<SGSHelperElement>;
using SGSGrid    = cubism::Grid<SGSBlock, aligned_allocator>;
using SGSGridMPI = cubism::GridMPI<SGSGrid>;

static inline SGSBlock* getSGSBlockPtr(
  SGSGridMPI*const grid, const int blockID) {
  assert(grid not_eq nullptr);
  const std::vector<BlockInfo>& vInfo = grid->getBlocksInfo();
  return (SGSBlock*) vInfo[blockID].ptrBlock;
}

template<bool readFromChi>
class KernelSGS_SSM
{
 private:
  const Real Cs;
  SGSGridMPI * const sgsGrid;

 public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W}};

  KernelSGS_SSM(SGSGridMPI*const _sgsGrid, const Real _Cs)
      : Cs(_Cs), sgsGrid(_sgsGrid) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o) const
  {
    const Real Cs2 = Cs*Cs, h = info.h_gridpoint;
    SGSBlock& t = * getSGSBlockPtr(sgsGrid, info.blockID);
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      SGSHelperElement& sgs = t(ix,iy,iz);

      const FluidElement &L =lab(ix,iy,iz);
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);

      const Real dudx = LE.u-LW.u, dvdx = LE.v-LW.v, dwdx = LE.w-LW.w;
      const Real dudy = LN.u-LS.u, dvdy = LN.v-LS.v, dwdy = LN.w-LS.w;
      const Real dudz = LB.u-LF.u, dvdz = LB.v-LF.v, dwdz = LB.w-LF.w;

      const Real shear = std::sqrt( 2*dudx*dudx + 2*dvdy*dvdy + 2*dwdz*dwdz
                                   + (dudy+dvdx)*(dudy+dvdx)
                                   + (dudz+dwdx)*(dudz+dwdx)
                                   + (dwdy+dvdz)*(dwdy+dvdz) ) / (2*h);
      const Real elemCs2 = readFromChi? L.chi : Cs2;
      sgs.nu = elemCs2 * h*h * shear;
      sgs.duD = (LN.u+LS.u + LE.u+LW.u + LF.u+LB.u - L.u*6)/(h*h);
      sgs.dvD = (LN.v+LS.v + LE.v+LW.v + LF.v+LB.v - L.v*6)/(h*h);
      sgs.dwD = (LN.w+LS.w + LE.w+LW.w + LF.w+LB.w - L.w*6)/(h*h);

      o(ix, iy, iz).tmpU = sgs.nu;
      if(!readFromChi) o(ix, iy, iz).chi = Cs2;
    }
  }
};

class KernelSGS_nonUniform
{
 private:
  const double dt;
  //const Real* const uInf;
  const bool bSGS_RL;

 public:
  const std::array<int, 3> stencil_start = {-1, -1, -1};
  const std::array<int, 3> stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W}};

  KernelSGS_nonUniform(double _dt, const Real* const _uInf, const bool _bSGS_RL)
      : dt(_dt), bSGS_RL(_bSGS_RL) {}

  ~KernelSGS_nonUniform() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o) const
  {
    const double Cs2 = 0.20*0.20;
    const BlkCoeffX & c1x = o.fd_cx.first, & c2x = o.fd_cx.second;
    const BlkCoeffY & c1y = o.fd_cy.first, & c2y = o.fd_cy.second;
    const BlkCoeffZ & c1z = o.fd_cz.first, & c2z = o.fd_cz.second;
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      Real h[3]; info.spacing(h, ix, iy, iz);
      const FluidElement &L =lab(ix,iy,iz);
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);

      const Real d1udx1 = __FD_2ND(ix, c1x, LW.u, L.u, LE.u);
      const Real d1udy1 = __FD_2ND(iy, c1y, LS.u, L.u, LN.u);
      const Real d1udz1 = __FD_2ND(iz, c1z, LF.u, L.u, LB.u);
      const Real d1vdx1 = __FD_2ND(ix, c1x, LW.v, L.v, LE.v);
      const Real d1vdy1 = __FD_2ND(iy, c1y, LS.v, L.v, LN.v);
      const Real d1vdz1 = __FD_2ND(iz, c1z, LF.v, L.v, LB.v);
      const Real d1wdx1 = __FD_2ND(ix, c1x, LW.w, L.w, LE.w);
      const Real d1wdy1 = __FD_2ND(iy, c1y, LS.w, L.w, LN.w);
      const Real d1wdz1 = __FD_2ND(iz, c1z, LF.w, L.w, LB.w);

      const Real d2udx2 = __FD_2ND(ix, c2x, LW.u, L.u, LE.u);
      const Real d2udy2 = __FD_2ND(iy, c2y, LS.u, L.u, LN.u);
      const Real d2udz2 = __FD_2ND(iz, c2z, LF.u, L.u, LB.u);
      const Real d2vdx2 = __FD_2ND(ix, c2x, LW.v, L.v, LE.v);
      const Real d2vdy2 = __FD_2ND(iy, c2y, LS.v, L.v, LN.v);
      const Real d2vdz2 = __FD_2ND(iz, c2z, LF.v, L.v, LB.v);
      const Real d2wdx2 = __FD_2ND(ix, c2x, LW.w, L.w, LE.w);
      const Real d2wdy2 = __FD_2ND(iy, c2y, LS.w, L.w, LN.w);
      const Real d2wdz2 = __FD_2ND(iz, c2z, LF.w, L.w, LB.w);
      const Real duD = d2udx2 + d2udy2 + d2udz2;
      const Real dvD = d2vdx2 + d2vdy2 + d2vdz2;
      const Real dwD = d2wdx2 + d2wdy2 + d2wdz2;
      const Real S = std::sqrt( 2*d1udx1*d1udx1
                               +2*d1vdy1*d1vdy1
                               +2*d1wdz1*d1wdz1
                               +(d1udy1+d1vdx1)*(d1udy1+d1vdx1)
                               +(d1udz1+d1wdx1)*(d1udz1+d1wdx1)
                               +(d1wdy1+d1vdz1)*(d1wdy1+d1vdz1));

      const Real facD = bSGS_RL ? L.chi * std::pow(h[0]*h[1]*h[2], 2.0 / 3) * S
                                :   Cs2 * std::pow(h[0]*h[1]*h[2], 2.0 / 3) * S;

      o(ix, iy, iz).tmpU = L.u + dt * facD * duD;
      o(ix, iy, iz).tmpV = L.v + dt * facD * dvD;
      o(ix, iy, iz).tmpW = L.w + dt * facD * dwD;
    }
  }
};

inline  Real facFilter(const int i, const int j, const int k)
{
  if (abs(i)+abs(j)+abs(k) == 3)         // Corner cells
    return 1.0/64;
  else if (abs(i)+abs(j)+abs(k) == 2)    // Side-Corner cells
    return 2.0/64;
  else if (abs(i)+abs(j)+abs(k) == 1)    // Side cells
    return 4.0/64;
  else if (abs(i)+abs(j)+abs(k) == 0)    // Center cells
    return 8.0/64;
  else // assert(false);
  return 0;
}

struct filterFluidElement
{
  Real u  = 0., v  = 0., w  = 0.;
  Real uu = 0., uv = 0., uw = 0.;
  Real vv = 0., vw = 0., ww = 0.;

  Real shear = 0.;
  Real S_xx = 0., S_xy = 0., S_xz = 0.;
  Real S_yy = 0., S_yz = 0., S_zz = 0.;
  Real shear_S_xx = 0., shear_S_xy = 0., shear_S_xz = 0.;
  Real shear_S_yy = 0., shear_S_yz = 0., shear_S_zz = 0.;

  filterFluidElement(Lab& lab, const int ix, const int iy, const int iz, const Real h)
  {
    for (int i = -1; i < 2; ++i)
    for (int j = -1; j < 2; ++j)
    for (int k = -1; k < 2; ++k)
    {
      const Real f = facFilter(i,j,k);
      const FluidElement & L = lab(ix+i, iy+j, iz+k);

      u  += f*L.u;     v  += f*L.v;     w  += f*L.w;
      uu += f*L.u*L.u; uv += f*L.u*L.v; uw += f*L.u*L.w;
      vv += f*L.v*L.v; vw += f*L.v*L.w; ww += f*L.w*L.w;

      const FluidElement &LW=lab(ix+i-1, iy+j,   iz+k  );
      const FluidElement &LE=lab(ix+i+1, iy+j,   iz+k  );
      const FluidElement &LS=lab(ix+i,   iy+j-1, iz+k  );
      const FluidElement &LN=lab(ix+i,   iy+j+1, iz+k  );
      const FluidElement &LF=lab(ix+i,   iy+j,   iz+k-1);
      const FluidElement &LB=lab(ix+i,   iy+k,   iz+k+1);

      const Real dudx = LE.u-LW.u, dvdx = LE.v-LW.v, dwdx = LE.w-LW.w;
      const Real dudy = LN.u-LS.u, dvdy = LN.v-LS.v, dwdy = LN.w-LS.w;
      const Real dudz = LB.u-LF.u, dvdz = LB.v-LF.v, dwdz = LB.w-LF.w;

      const Real shear_g = std::sqrt( 2 * (dudx*dudx) +
                                      2 * (dvdy*dvdy) +
                                      2 * (dwdz*dwdz) +
                                      std::pow(dudy+dvdx, 2) +
                                      std::pow(dudz+dwdx, 2) +
                                      std::pow(dwdy+dvdz, 2) ) / (2*h);

      shear += f * shear_g;

      shear_S_xx += f * shear_g *  dudx         / (2*h);
      shear_S_xy += f * shear_g * (dudy + dvdx) / (2*2*h);
      shear_S_xz += f * shear_g * (dudz + dwdx) / (2*2*h);
      shear_S_yy += f * shear_g *  dvdy         / (2*h);
      shear_S_yz += f * shear_g * (dwdy + dvdz) / (2*2*h);
      shear_S_zz += f * shear_g *  dwdz         / (2*h);

      S_xx += f *  dudx         / (2*h);
      S_xy += f * (dudy + dvdx) / (2*2*h);
      S_xz += f * (dudz + dwdx) / (2*2*h);
      S_yy += f *  dvdy         / (2*h);
      S_yz += f * (dwdy + dvdz) / (2*2*h);
      S_zz += f *  dwdz         / (2*h);
    }
  }
};

class KernelSGS_DSM
{
 private:
  SGSGridMPI * const sgsGrid;

 public:
  const std::array<int, 3> stencil_start = {-3, -3, -3};
  const std::array<int, 3> stencil_end = {4, 4, 4};
  const StencilInfo stencil{-3,-3,-3, 4,4,4, true, {FE_U,FE_V,FE_W}};
  #ifdef DSM_LILLY
  mutable Real mean_l_dot_m = 0;
  mutable Real mean_m_dot_m = 0;
  #endif

  KernelSGS_DSM(SGSGridMPI * const _sgsGrid)
      : sgsGrid(_sgsGrid) {}

  ~KernelSGS_DSM() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o) const
  {
    const Real h = info.h_gridpoint, mFac = 2 * h * h;
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {

      const filterFluidElement L_f(lab,ix,iy,iz, h);

      const Real m_xx = mFac * (L_f.shear_S_xx - 4 * L_f.shear * L_f.S_xx);
      const Real m_xy = mFac * (L_f.shear_S_xy - 4 * L_f.shear * L_f.S_xy);
      const Real m_xz = mFac * (L_f.shear_S_xz - 4 * L_f.shear * L_f.S_xz);
      const Real m_yy = mFac * (L_f.shear_S_yy - 4 * L_f.shear * L_f.S_yy);
      const Real m_yz = mFac * (L_f.shear_S_yz - 4 * L_f.shear * L_f.S_yz);
      const Real m_zz = mFac * (L_f.shear_S_zz - 4 * L_f.shear * L_f.S_zz);

      const Real traceTerm = 1.0/3 * (L_f.uu + L_f.vv + L_f.ww
                              - L_f.u * L_f.u - L_f.v * L_f.v - L_f.w * L_f.w);
      const Real l_xx = L_f.uu - L_f.u * L_f.u - traceTerm;
      const Real l_xy = L_f.uv - L_f.u * L_f.v;
      const Real l_xz = L_f.uw - L_f.u * L_f.w;
      const Real l_yy = L_f.vv - L_f.v * L_f.v - traceTerm;
      const Real l_yz = L_f.vw - L_f.v * L_f.w;
      const Real l_zz = L_f.ww - L_f.w * L_f.w - traceTerm;

      const Real l_dot_m = l_xx * m_xx + l_yy * m_yy + l_zz * m_zz +
                      2 * (l_xy * m_xy + l_xz * m_xz + l_yz * m_yz);

      const Real m_dot_m = m_xx * m_xx + m_yy * m_yy + m_zz * m_zz +
                      2 * (m_xy * m_xy + m_xz * m_xz + m_yz * m_yz);

      o(ix,iy,iz).tmpV = l_dot_m;
      o(ix,iy,iz).tmpW = m_dot_m;
      #ifdef DSM_LILLY
        mean_l_dot_m += l_dot_m;
        mean_m_dot_m += m_dot_m;
      #endif
    }
  }
};

class KernelSGS_DSM_avg
{
 private:
  SGSGridMPI * const sgsGrid;

 public:
  const std::array<int, 3> stencil_start = {-1, -1, -1};
  const std::array<int, 3> stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, true, {FE_U,FE_V,FE_W,
                                                    FE_TMPV,FE_TMPW}};

  KernelSGS_DSM_avg(SGSGridMPI * const _sgsGrid)
      : sgsGrid(_sgsGrid) {}

  ~KernelSGS_DSM_avg() {}
  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o) const
  {
    const Real h = info.h_gridpoint;
    SGSBlock& t = * getSGSBlockPtr(sgsGrid, info.blockID);
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      SGSHelperElement& sgs = t(ix,iy,iz);

      // To be modeled by sgs
      const FluidElement &L =lab(ix,iy,iz);
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);

      const Real d1udx1= LE.u-LW.u, d1vdx1= LE.v-LW.v, d1wdx1= LE.w-LW.w;
      const Real d1udy1= LN.u-LS.u, d1vdy1= LN.v-LS.v, d1wdy1= LN.w-LS.w;
      const Real d1udz1= LB.u-LF.u, d1vdz1= LB.v-LF.v, d1wdz1= LB.w-LF.w;
      const Real shear = std::sqrt( 2*d1udx1*d1udx1
                                   +2*d1vdy1*d1vdy1
                                   +2*d1wdz1*d1wdz1
                                   +(d1udy1+d1vdx1)*(d1udy1+d1vdx1)
                                   +(d1udz1+d1wdx1)*(d1udz1+d1wdx1)
                                   +(d1wdy1+d1vdz1)*(d1wdy1+d1vdz1))/(2*h);

      #ifndef DSM_LOCAL
      Real l_dot_m = 0.0, m_dot_m = 0.0;
      for (int i=-1; i<2; ++i)
      for (int j=-1; j<2; ++j)
      for (int k=-1; k<2; ++k) {
        l_dot_m += facFilter(i,j,k) * lab(ix+i, iy+j, iz+k).tmpV;
        m_dot_m += facFilter(i,j,k) * lab(ix+i, iy+j, iz+k).tmpW;
      }
      const Real hat = l_dot_m            / std::max(m_dot_m,            EPS);
      const Real loc = lab(ix,iy,iz).tmpV / std::max(lab(ix,iy,iz).tmpW, EPS);
      const Real Cs2 = std::max({hat, loc, (Real) 0});
      #else
      const Real Cs2 = std::max(lab(ix,iy,iz).tmpV, EPS)
                     / std::max(lab(ix,iy,iz).tmpW, EPS);
      #endif

      sgs.nu = Cs2 * h*h * shear;
      sgs.duD = (LN.u+LS.u + LE.u+LW.u + LF.u+LB.u - L.u*6)/(h*h);
      sgs.dvD = (LN.v+LS.v + LE.v+LW.v + LF.v+LB.v - L.v*6)/(h*h);
      sgs.dwD = (LN.w+LS.w + LE.w+LW.w + LF.w+LB.w - L.w*6)/(h*h);
      o(ix,iy,iz).tmpU = sgs.nu;
      o(ix,iy,iz).chi = Cs2;
    }
  }
};

class KernelSGS_gradNu
{
 private:
  SGSGridMPI * const sgsGrid;

 public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2,2,2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W,FE_TMPU}};

  KernelSGS_gradNu(SGSGridMPI * const _sgsGrid) : sgsGrid(_sgsGrid) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o) const
  {
    const Real h = info.h_gridpoint, F = 1/(2*h);
    SGSBlock& t = * getSGSBlockPtr(sgsGrid, info.blockID);
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      const FluidElement &LW = lab(ix-1,iy,iz), &LE = lab(ix+1,iy,iz);
      const FluidElement &LS = lab(ix,iy-1,iz), &LN = lab(ix,iy+1,iz);
      const FluidElement &LF = lab(ix,iy,iz-1), &LB = lab(ix,iy,iz+1);

      const Real dudx = LE.u-LW.u, dvdx = LE.v-LW.v, dwdx = LE.w-LW.w;
      const Real dudy = LN.u-LS.u, dvdy = LN.v-LS.v, dwdy = LN.w-LS.w;
      const Real dudz = LB.u-LF.u, dvdz = LB.v-LF.v, dwdz = LB.w-LF.w;
      const Real dnudx = LE.tmpU-LW.tmpU;
      const Real dnudy = LN.tmpU-LS.tmpU;
      const Real dnudz = LB.tmpU-LF.tmpU;
      const Real DnuSx = dudx*dnudx + dnudy*(dudy+dvdx)/2 + dnudz*(dudz+dwdx)/2;
      const Real DnuSy = dnudx*(dudy+dvdx)/2 + dnudy*dvdy + dnudz*(dvdz+dwdy)/2;
      const Real DnuSz = dnudx*(dudz+dwdx)/2 + dnudy*(dudy+dvdx)/2 + dnudz*dwdz;
      // double F factor because of fdiff u multiplied by f diff nu
      t(ix,iy,iz).duD = t(ix,iy,iz).nu * t(ix,iy,iz).duD + 2 * F*F * DnuSx;
      t(ix,iy,iz).dvD = t(ix,iy,iz).nu * t(ix,iy,iz).dvD + 2 * F*F * DnuSy;
      t(ix,iy,iz).dwD = t(ix,iy,iz).nu * t(ix,iy,iz).dwD + 2 * F*F * DnuSz;
    }
  }
};

class KernelSGS_apply
{
  const Real dt;
  SGSGridMPI * const sgsGrid;

public:
  KernelSGS_apply(Real _dt, SGSGridMPI*const _sgs) : dt(_dt), sgsGrid(_sgs) {}

  void operator()(const BlockInfo& info, FluidBlock& o) const
  {
    const SGSBlock& t = * getSGSBlockPtr(sgsGrid, info.blockID);
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      o(ix,iy,iz).u += dt * t(ix,iy,iz).duD;
      o(ix,iy,iz).v += dt * t(ix,iy,iz).dvD;
      o(ix,iy,iz).w += dt * t(ix,iy,iz).dwD;
    }
  }
};

class KernelSGS_applyAndAnalyze
{
  const Real dt;
  SGSGridMPI * const sgsGrid;

  int CStoBinID(const Real CS2) const
  {
    const int signedID = (CS2 - minCS2) * nBins / (maxCS2 - minCS2);
    //printf("%d %e\n", signedID, CS2);
    return std::max((int) 0, std::min(signedID, nBins-1));
  }

public:

  static constexpr Real maxCS2 = 0.09;
  static constexpr Real minCS2 = 0;
  static constexpr int nBins = 90;
  Real cs2_sum = 0.0, cs2_sum2 = 0.0, nuSGS_sum = 0.0, nuSGS_sum2 = 0.0;
  int histogramCS2[nBins] = {0};

  KernelSGS_applyAndAnalyze(Real _dt, SGSGridMPI*const _sgs) : dt(_dt), sgsGrid(_sgs) {}

  void operator()(const BlockInfo& info, FluidBlock& o)
  {
    const SGSBlock& t = * getSGSBlockPtr(sgsGrid, info.blockID);
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      o(ix,iy,iz).u += dt * t(ix,iy,iz).duD;
      o(ix,iy,iz).v += dt * t(ix,iy,iz).dvD;
      o(ix,iy,iz).w += dt * t(ix,iy,iz).dwD;

      cs2_sum += o(ix,iy,iz).chi;
      cs2_sum2 += o(ix,iy,iz).chi * o(ix,iy,iz).chi;
      nuSGS_sum += t(ix,iy,iz).nu;
      nuSGS_sum2 += t(ix,iy,iz).nu * t(ix,iy,iz).nu;
      const int indCS = CStoBinID(o(ix,iy,iz).chi);
      ++histogramCS2[indCS];
      //printf("%d %d %e\n", indCS, histogramCS2[indCS], o(ix,iy,iz).chi);
    }
  }
};

SGS::SGS(SimulationData& s) : Operator(s) {
  _sgsGrid = new SGSGridMPI(sim.nprocsx, sim.nprocsy, sim.nprocsz,
    sim.local_bpdx, sim.local_bpdy, sim.local_bpdz, sim.maxextent, sim.app_comm);
}

SGS::~SGS() {
  delete (SGSGridMPI*) _sgsGrid;
}

void SGS::operator()(const double dt)
{
  SGSGridMPI * sgsGrid = (SGSGridMPI*) _sgsGrid;
  sim.startProfiler("SGS Kernel");
  if(sim.bUseStretchedGrid) {
    printf("ERROR: SGS model not implemented with non uniform grid.\n");
    fflush(0); abort();
    //const KernelSGS_nonUniform sgs(dt, sim.uinf.data());
    //compute<KernelSGS_nonUniform>(sgs);
  } else {
    if (sim.sgs=="DSM" or sim.cs < 0) { // Dynamic Smagorinsky Model
      #ifndef DSM_LILLY
        const KernelSGS_DSM computeCs(sgsGrid);
        compute(computeCs);

        const KernelSGS_DSM_avg averageCs(sgsGrid);
        compute(averageCs);
      #else
        const int nthreads = omp_get_max_threads();
        std::vector<KernelSGS_DSM*> K(nthreads, nullptr);
        for(int i=0; i<nthreads; ++i) K[i] = new KernelSGS_DSM(sgsGrid);
        compute(K);
        double mean[2];
        for(int i=0; i<nthreads; ++i) {
          mean[0] += K[i]->mean_l_dot_m;
          mean[1] += K[i]->mean_m_dot_m;
          delete K[i];
        }
        MPI_Allreduce(MPI_IN_PLACE, mean, 2, MPI_DOUBLE, MPI_SUM, sim.app_comm);
        const Real CS2 = mean[0] / std::max(mean[1], EPS); // prevent nan
        const Real Cs = std::sqrt(std::max(CS2, EPS));     // prevent nan
        const KernelSGS_SSM<false> applyCs(sgsGrid, Cs);
        compute(applyCs);
      #endif
    }
    else if (sim.sgs=="SSM") { // Standard Smagorinsky Model
      const KernelSGS_SSM<false> K(sgsGrid, sim.cs);
      compute(K);
    }
    else if (sim.sgs=="RLSM") { // RL Smagorinsky Model
      const KernelSGS_SSM<true> K(sgsGrid, sim.cs);
      compute(K);
    }
  }

  using K_t = KernelSGS_gradNu;
  const K_t K(sgsGrid);
  compute<K_t>(K);

  if (sim.timeAnalysis>0 && (sim.time+dt) >= sim.nextAnalysisTime)
  {
    static constexpr int nBins = KernelSGS_applyAndAnalyze::nBins;
    int histogram[nBins] = {0};
    double reduction[4] = {(Real) 0, (Real) 0, (Real) 0, (Real) 0};
    #pragma omp parallel reduction(+ : reduction[:4], histogram[:nBins])
    {
      KernelSGS_applyAndAnalyze kernel(dt, sgsGrid);
      #pragma omp for schedule(static)
      for (size_t i=0; i<vInfo.size(); i++)
        kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);

      reduction[0] += kernel.cs2_sum;
      reduction[1] += kernel.cs2_sum2;
      reduction[2] += kernel.nuSGS_sum;
      reduction[3] += kernel.nuSGS_sum2;
      for (int i=0; i<nBins; ++i) histogram[i] += kernel.histogramCS2[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, reduction, 4, MPI_DOUBLE, MPI_SUM, sim.app_comm);
    MPI_Allreduce(MPI_IN_PLACE, histogram, nBins, MPI_INT, MPI_SUM, sim.app_comm);
    const size_t normalize =  FluidBlock::sizeX * (size_t) sim.bpdx
                            * FluidBlock::sizeY * (size_t) sim.bpdy
                            * FluidBlock::sizeZ * (size_t) sim.bpdz;
    const Real meanCS2 = reduction[0] / normalize;
    const Real meanNUS = reduction[2] / normalize;
    const Real varCS2 = reduction[1] / normalize - meanCS2 * meanCS2;
    const Real varNUS = reduction[3] / normalize - meanNUS * meanNUS;
    sim.cs2mean   = meanCS2; sim.cs2stdev   = std::sqrt(varCS2);
    sim.nuSgsMean = meanNUS; sim.nuSgsStdev = std::sqrt(varNUS);
    if(sim.rank==0 and not sim.muteAll) {
      std::vector<double> buf{meanCS2, meanNUS, varCS2, varNUS};
      buf.reserve(buf.size() + nBins);
      //for (int i=0; i<nBins; ++i) printf("%d\n", histogram[i]);
      for (int i = 0; i < nBins; ++i)
          buf.push_back(histogram[i] / (double) normalize);
      FILE * pFile = fopen ("sgsAnalysis.raw", "ab");
      fwrite (buf.data(), sizeof(double), buf.size(), pFile);
      fflush(pFile); fclose(pFile);
    }
  }
  else
  {
    const KernelSGS_apply kernel(dt, sgsGrid);
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<vInfo.size(); i++)
        kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
  }

  sim.stopProfiler();

  check("SGS");
}

CubismUP_3D_NAMESPACE_END
