//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Operator.h"
#include "SGS_RL.h"
#include "smarties.h"
#include "../spectralOperators/SpectralManip.h"
#include "../spectralOperators/HITtargetData.h"

#include <functional>
CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

static constexpr int agentLocInBlock = CUP_BLOCK_SIZE/2;

#define GIVE_LOCL_STATEVARS
#define GIVE_GRID_STATEVARS

struct ActionInterpolator
{
  const int NX, NY, NZ;
  static constexpr int NB = CUP_BLOCK_SIZE;

  std::vector<std::vector<std::vector<double>>> actions =
    std::vector<std::vector<std::vector<double>>> (
      NZ, std::vector<std::vector<double>>(NY, std::vector<double>(NX, 0) ) );

  ActionInterpolator(int _NX, int _NY, int _NZ) : NX(_NX), NY(_NY), NZ(_NZ)
  {
  }

  double operator()(const int bix, const int biy, const int biz,
    const int ix, const int iy, const int iz) const
  {
    // linear interpolation betwen element's block (bix, biy, biz) and second
    // nearest. figure out which from element's index (ix, iy, iz) in block:
    const int nbix = ix < NB/2 ? bix - 1 : bix + 1;
    const int nbiy = iy < NB/2 ? biy - 1 : biy + 1;
    const int nbiz = iz < NB/2 ? biz - 1 : biz + 1;
    // distance from second nearest block along its direction:
    const Real dist_nbix = ix < NB/2 ? NB/2 + ix + 0.5 : 3*NB/2 - ix - 0.5;
    const Real dist_nbiy = iy < NB/2 ? NB/2 + iy + 0.5 : 3*NB/2 - iy - 0.5;
    const Real dist_nbiz = iz < NB/2 ? NB/2 + iz + 0.5 : 3*NB/2 - iz - 0.5;
    // distance from block's center:
    const Real dist_bix = std::fabs(ix + 0.5 - NB/2);
    const Real dist_biy = std::fabs(iy + 0.5 - NB/2);
    const Real dist_biz = std::fabs(iz + 0.5 - NB/2);

    double weighted_sum_act = 0;
    double sum_acts_weights = 0;
    for(int z = 0; z < 2; ++z) // 0 is current block, 1 is nearest along z, y, x
      for(int y = 0; y < 2; ++y)
        for(int x = 0; x < 2; ++x) {
          const Real distx = x? dist_nbix : dist_bix;
          const Real disty = y? dist_nbiy : dist_biy;
          const Real distz = z? dist_nbiz : dist_biz;
          const Real act = action(x? nbix : bix, y? nbiy : biy, z? nbiz : biz);
          const Real dist = std::sqrt(distx*distx + disty*disty + distz*distz);
          const Real weight = std::max( (NB - dist)/NB, (Real) 0);
          weighted_sum_act += act * weight;
          sum_acts_weights += weight;
        }
    //return sum_acts_weights;
    return weighted_sum_act / std::max(sum_acts_weights, (double) 1e-16);
  }

  void set(const double act, const int bix, const int biy, const int biz)
  {
    //printf("action:%e\n", act);
    action(bix, biy, biz) = act;
  }

  const double & action(int bix, int biy, int biz) const
  {
    return actions[(biz+NZ) % NZ][(biy+NY) % NY][(bix+NX) % NX];
  }
  double & action(int bix, int biy, int biz)
  {
    return actions[(biz+NZ) % NZ][(biy+NY) % NY][(bix+NX) % NX];
  }
};

// Product of two symmetric matrices stored as 1D vectors with 6 elts {M_00, M_01, M_02,
//                                                                           M_11, M_12,
//                                                                                 M_22}
// Returns a symmetric matrix.
inline std::array<Real,6> symProd(const std::array<Real,6> & mat1,
                                  const std::array<Real,6> & mat2)
{
  return { mat1[0]*mat2[0] + mat1[1]*mat2[1] + mat1[2]*mat2[2],
           mat1[0]*mat2[1] + mat1[1]*mat2[3] + mat1[2]*mat2[4],
           mat1[0]*mat2[2] + mat1[1]*mat2[4] + mat1[2]*mat2[5],
           mat1[1]*mat2[1] + mat1[3]*mat2[3] + mat1[4]*mat2[4],
           mat1[1]*mat2[2] + mat1[3]*mat2[4] + mat1[4]*mat2[5],
           mat1[2]*mat2[2] + mat1[4]*mat2[4] + mat1[5]*mat2[5] };
}
// Product of two anti symmetric matrices stored as 1D vector with 3 elts (M_01, M_02, M_12)
// Returns a symmetric matrix.
inline std::array<Real,6> antiSymProd(const std::array<Real,3> & mat1,
                                      const std::array<Real,3> & mat2)
{
  return { - mat1[0]*mat2[0] - mat1[1]*mat2[1],
           - mat1[1]*mat2[2],
             mat1[0]*mat2[2],
           - mat1[0]*mat2[0] - mat1[2]*mat2[2],
           - mat1[0]*mat2[1],
           - mat1[1]*mat2[1] - mat1[2]*mat2[2] };
}
// Returns the Tr[mat1*mat2] with mat1 and mat2 symmetric matrices stored as 1D vector.
inline Real traceOfSymProd(const std::array<Real,6> & mat1,
                           const std::array<Real,6> & mat2)
{
  return mat1[0]*mat2[0] +   mat1[3]*mat2[3]  +   mat1[5]*mat2[5]
     + 2*mat1[1]*mat2[1] + 2*mat1[2]*mat2[2]  + 2*mat1[4]*mat2[4];
}


using rlApi_t = std::function<Real(const std::array<Real,13> &, const size_t,
                                   const size_t,const int,const int,const int)>;
using locRewF_t = std::function<void(const size_t blockID, Lab & lab)>;

inline Real sqrtDist(const Real val) {
  return val>=0? std::sqrt(val) : -std::sqrt(-val);
};
inline Real cbrtDist(const Real val) {
  return std::cbrt(val);
};
inline Real frthDist(const Real val) {
  return val>=0? std::sqrt(std::sqrt(val)) : -std::sqrt(std::sqrt(-val));
};

inline std::array<Real,6> popeInvariants(
  const Real d1udx1, const Real d1vdx1, const Real d1wdx1,
  const Real d1udy1, const Real d1vdy1, const Real d1wdy1,
  const Real d1udz1, const Real d1vdz1, const Real d1wdz1)
{
  const std::array<Real,6> S = { d1udx1, (d1vdx1 + d1udy1)/2,
    (d1wdx1 + d1udz1)/2, d1vdy1, (d1wdy1 + d1vdz1)/2, d1wdz1 };
  const std::array<Real,3> R = {
    (d1vdx1 - d1udy1)/2, (d1wdx1 - d1udz1)/2, (d1wdy1 - d1vdz1)/2 };
  const std::array<Real,6> S2  = symProd(S, S);
  const std::array<Real,6> R2  = antiSymProd(R, R);
  //const std::vector<Real> R2S = symProd(R2, S);
  return {         ( S[0] +  S[3] +  S[5]),   // Tr(S)
           sqrtDist(S2[0] + S2[3] + S2[5]),   // Tr(S^2)
           sqrtDist(R2[0] + R2[3] + R2[5]),   // Tr(R^2)
           cbrtDist(traceOfSymProd(S2, S)),   // Tr(S^3)
           cbrtDist(traceOfSymProd(R2, S)),   // Tr(R^2.S)
           frthDist(traceOfSymProd(R2,S2)) }; // Tr(R^2.S^2)
}

inline std::array<Real,3> mainMatInvariants(
  const Real xx, const Real xy, const Real xz,
  const Real yx, const Real yy, const Real yz,
  const Real zx, const Real zy, const Real zz)
{
  const Real I1 = xx + yy + zz; // Tr(Mat)
  // ( Tr(Mat)^2 - Tr(Mat^2) ) / 2:
  const Real I2 = xx*yy + yy*zz + xx*zz - xy*yx - yz*zy - xz*zx;
  // Det(Mat):
  const Real I3 = xy*yz*zx + xz*yx*zy + xx*yy*zz
                - xz*yy*zx - xx*yz*zy - xy*yx*zz;
  return {I1, sqrtDist(I2), cbrtDist(I3)};
}

template <typename Lab>
inline std::array<Real, 13> getState_uniform(Lab& lab, const Real h,
      const Real facVel, const Real facGrad, const Real facLap,
      const int ix, const int iy, const int iz)
{
  const FluidElement &L  = lab(ix, iy, iz);
  const FluidElement &LW = lab(ix - 1, iy, iz), &LE = lab(ix + 1, iy, iz);
  const FluidElement &LS = lab(ix, iy - 1, iz), &LN = lab(ix, iy + 1, iz);
  const FluidElement &LF = lab(ix, iy, iz - 1), &LB = lab(ix, iy, iz + 1);

  const Real d1udx = facGrad*(LE.u-LW.u), d2udx = facLap*(LE.u+LW.u-L.u*2);
  const Real d1vdx = facGrad*(LE.v-LW.v), d2vdx = facLap*(LE.v+LW.v-L.v*2);
  const Real d1wdx = facGrad*(LE.w-LW.w), d2wdx = facLap*(LE.w+LW.w-L.w*2);
  const Real d1udy = facGrad*(LN.u-LS.u), d2udy = facLap*(LN.u+LS.u-L.u*2);
  const Real d1vdy = facGrad*(LN.v-LS.v), d2vdy = facLap*(LN.v+LS.v-L.v*2);
  const Real d1wdy = facGrad*(LN.w-LS.w), d2wdy = facLap*(LN.w+LS.w-L.w*2);
  const Real d1udz = facGrad*(LB.u-LF.u), d2udz = facLap*(LB.u+LF.u-L.u*2);
  const Real d1vdz = facGrad*(LB.v-LF.v), d2vdz = facLap*(LB.v+LF.v-L.v*2);
  const Real d1wdz = facGrad*(LB.w-LF.w), d2wdz = facLap*(LB.w+LF.w-L.w*2);
  const Real S0 = facVel * std::sqrt(L.u*L.u + L.v*L.v + L.w*L.w);
  const std::array<double,6> S1 = popeInvariants(d1udx, d1vdx, d1wdx,
                                                 d1udy, d1vdy, d1wdy,
                                                 d1udz, d1vdz, d1wdz);
  const std::array<double,6> S2 = popeInvariants(d2udx, d2vdx, d2wdx,
                                                 d2udy, d2vdy, d2wdy,
                                                 d2udz, d2vdz, d2wdz);
  return {S0, S1[0], S1[1], S1[2], S1[3], S1[4], S1[5],
              S2[0], S2[1], S2[2], S2[3], S2[4], S2[5]};
}

struct KernelSGS_RL
{
  const rlApi_t & sendStateRecvAct;
  ActionInterpolator & actInterp;
  const HITstatistics & stats;
  const Real scaleVel, scaleGrad, scaleLap;

  const StencilInfo stencil{-1,-1,-1, 2, 2, 2, false, {FE_U,FE_V,FE_W}};

  KernelSGS_RL(const rlApi_t& api,
        ActionInterpolator& interp, const HITstatistics& _stats,
        const Real _facVel, const Real _facGrad, const Real _facLap) :
        sendStateRecvAct(api), actInterp(interp), stats(_stats),
        scaleVel(_facVel), scaleGrad(_facGrad), scaleLap(_facLap) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& o) const
  {
    // FD coefficients for first and second derivative
    const Real h = info.h_gridpoint;
    const Real facV = scaleVel, facG = scaleGrad/(2*h), facL = scaleLap/(h*h);
    const size_t thrID = omp_get_thread_num(), blockID = info.blockID;

    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      const auto state = getState_uniform(lab, h, facV, facG, facL, ix, iy, iz);
      // LES coef can be stored in chi as long as we do not have obstacles
      // otherwise we will have to figure out smth
      // we could compute a local reward here, place as second arg
      o(ix,iy,iz).chi = sendStateRecvAct(state, blockID, thrID, ix,iy,iz);
    }
  }

  void state_center(const BlockInfo& info)
  {
    FluidBlock & o = * (FluidBlock *) info.ptrBlock;
    // FD coefficients for first and second derivative
    const Real h = info.h_gridpoint;
    const Real facV = scaleVel, facG = scaleGrad/(2*h), facL = scaleLap/(h*h);
    const size_t thrID = omp_get_thread_num(), blockID = info.blockID;
    const int idx = CUP_BLOCK_SIZE/2 - 1, ipx = CUP_BLOCK_SIZE/2;
    std::array<Real, 13> avgState = {0.};
    const double factor = 1.0 / 8;
    for (int iz = idx; iz <= ipx; ++iz)
    for (int iy = idx; iy <= ipx; ++iy)
    for (int ix = idx; ix <= ipx; ++ix) {
      const auto state = getState_uniform(o, h, facV, facG, facL, ix, iy, iz);
      for (int k = 0; k < 13; ++k) avgState[k] += factor * state[k];
      // LES coef can be stored in chi as long as we do not have obstacles
      // otherwise we will have to figure out smth
      // we could compute a local reward here, place as second arg
    }
    actInterp.set(sendStateRecvAct(avgState, blockID, thrID, idx,idx,idx),
      info.index[0], info.index[1], info.index[2]);
  }

  void apply_actions(const BlockInfo & i) const
  {
    FluidBlock & o = * (FluidBlock *) i.ptrBlock;
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix)
      o(ix,iy,iz).chi = actInterp(i.index[0], i.index[1], i.index[2], ix,iy,iz);
  }
};

struct KernelGermanoError
{
  // In principle, SGS closure (a prescription for C_s) can be derived from
  // properties of the resolved flow. C_s should be independant of the chosen
  // filter scale h as long as h is in the inertial range of the turbulent flow.
  // If you consider 2 filters with width H>h, then the residual-stress tensor :
  //        \tau(H) = Filter_H(\tau(h)) + L(h, H)                                           (1)
  // where the Leonard tensor L(h, H) is the residual-stress tensor associated
  // to the vel fluctiation in the intermediate range of scales h < ... < H :
  //        L_ij(h, H) = Filter_H( u_i(h) u_j(h)) - Filter_H( u_i(h) ) Filter_H( u_j(h) )   (2)
  // In LES, (2) can evalated using the resolved flow only whereas \tau(H)
  // and \tau(h) are obtained from a SGS closure. A good SGS closure should
  // satisfy the Germano identity (1) for any test filter H, if h and H are
  // in the intertial range.
  //
  // Returns the tensor components of [(2) - L_ij]. A perfect closure model
  // would return only zeros for any test filter H.
  const Real facVel;
  const HITstatistics & stats;
  std::vector<double> & localRewards;
  //const StencilInfo stencil{-1,-1,-1, 2, 2, 2, true, {FE_CHI,FE_U,FE_V,FE_W}};
  const StencilInfo stencil = StencilInfo(-2,-2,-2, 3,3,3, true, {FE_U,FE_V,FE_W});

  static Real facFilter(const int i, const int j, const int k)
  {
    if (std::abs(i) + std::abs(j) + std::abs(k) == 3)         // Corner cells
      return 1.0/64;
    else if (std::abs(i) + std::abs(j) + std::abs(k) == 2)    // Side-Corner cells
      return 2.0/64;
    else if (std::abs(i) + std::abs(j) + std::abs(k) == 1)    // Side cells
      return 4.0/64;
    else if (std::abs(i) + std::abs(j) + std::abs(k) == 0)    // Center cells
      return 8.0/64;
    else return 0;
  }

  // FilteredQuantities is a container that will be used to evaluate (1) with a
  // linear test filter with H = 2 * h (people rather use Gaussian filters).
  struct FilteredQuantities
  {
    // Filter_H( u_i(h) ) : filtered velocity components.
    Real u  = 0.0, v  = 0.0, w  = 0.0;
    // Filter_H( u_i(h) u_j(h) ) : product of velocity components filtered.
    Real uu = 0.0, uv = 0.0, uw = 0.0, vv = 0.0, vw = 0.0, ww = 0.0;
    // Filtered_H( S_ij(h) ) : used to compute \tau(H).
    Real S_xx = 0.0, S_xy = 0.0, S_xz = 0.0, S_yy = 0.0, S_yz = 0.0, S_zz = 0.0;
    // Filter_H ( tau(h) ).
    Real tau_xx = 0.0, tau_xy = 0.0, tau_xz = 0.0;
    Real tau_yy = 0.0, tau_yz = 0.0, tau_zz = 0.0;

    template <typename Lab>
    FilteredQuantities(Lab& lab, const int ix, const int iy, const int iz, const Real h)
    {
      for (int k=-1; k<2; ++k)
      for (int j=-1; j<2; ++j)
      for (int i=-1; i<2; ++i) {
        const Real f = facFilter(i,j,k);
        const auto & L = lab(ix+i, iy+j, iz+k);
        u  += f * L.u;     v  += f * L.v;     w  += f * L.w;
        uu += f * L.u*L.u; uv += f * L.u*L.v; uw += f * L.u*L.w;
        vv += f * L.v*L.v; vw += f * L.v*L.w; ww += f * L.w*L.w;
        const auto & LW=lab(ix+i-1, iy+j, iz+k), & LE=lab(ix+i+1, iy+j, iz+k);
        const auto & LS=lab(ix+i, iy+j-1, iz+k), & LN=lab(ix+i, iy+j+1, iz+k);
        const auto & LF=lab(ix+i, iy+j, iz+k-1), & LB=lab(ix+i, iy+k, iz+k+1);
        const Real dudx = LE.u-LW.u, dvdx = LE.v-LW.v, dwdx = LE.w-LW.w;
        const Real dudy = LN.u-LS.u, dvdy = LN.v-LS.v, dwdy = LN.w-LS.w;
        const Real dudz = LB.u-LF.u, dvdz = LB.v-LF.v, dwdz = LB.w-LF.w;
        S_xx += f * (dudx)        / (2*h);
        S_xy += f * (dudy + dvdx) / (4*h);
        S_xz += f * (dudz + dwdx) / (4*h);
        S_yy += f * (dvdy)        / (2*h);
        S_yz += f * (dvdz + dwdy) / (4*h);
        S_zz += f * (dwdz)        / (2*h);
        const Real shear = std::sqrt( 2 * (dudx*dudx) +
                                      2 * (dvdy*dvdy) +
                                      2 * (dwdz*dwdz) +
                                      pow2(dudy+dvdx) +
                                      pow2(dudz+dwdx) +
                                      pow2(dwdy+dvdz) ) / (2*h);
        const Real tau_factor = - 2 * L.chi * shear * h * h;
        tau_xx += f * tau_factor * (dudx)        / (2*h);
        tau_xy += f * tau_factor * (dudy + dvdx) / (4*h);
        tau_xz += f * tau_factor * (dudz + dwdx) / (4*h);
        tau_yy += f * tau_factor * (dvdy)        / (2*h);
        tau_yz += f * tau_factor * (dwdy + dvdz) / (4*h);
        tau_zz += f * tau_factor * (dwdz)        / (2*h);
      }
      // Remove trace of Filter_H( \tau(h) )
      // NOTE: By construction, it should already be trace free.
      const Real tau_trace = (tau_xx + tau_yy + tau_zz)/3;
      tau_xx -= tau_trace;
      tau_yy -= tau_trace;
      tau_zz -= tau_trace;
    }
  };

  template <typename Lab>
  static Real germanoError(Lab & lab, const Real h, const Real scaleVel,
                           const int ix, const int iy, const int iz)
  {
    const Real H = 2*h;
    FilteredQuantities filter_H(lab, ix,iy,iz, h);
    const FluidElement & L = lab(ix, iy, iz);
    const Real shear_H = std::sqrt( 2 * pow2(filter_H.S_xx) +
                                    2 * pow2(filter_H.S_yy) +
                                    2 * pow2(filter_H.S_zz) +
                                    4 * pow2(filter_H.S_xy) +
                                    4 * pow2(filter_H.S_yz) +
                                    4 * pow2(filter_H.S_xz) );
    const Real L_trace = ( filter_H.uu + filter_H.vv + filter_H.ww
                          - filter_H.u * filter_H.u
                          - filter_H.v * filter_H.v
                          - filter_H.w * filter_H.w ) / 3;
    // Leonard tensor:
    const Real L_xx = filter_H.uu - filter_H.u * filter_H.u - L_trace;
    const Real L_xy = filter_H.uv - filter_H.u * filter_H.v;
    const Real L_xz = filter_H.uw - filter_H.u * filter_H.w;
    const Real L_yy = filter_H.vv - filter_H.v * filter_H.v - L_trace;
    const Real L_yz = filter_H.vw - filter_H.v * filter_H.w;
    const Real L_zz = filter_H.ww - filter_H.w * filter_H.w - L_trace;
    const Real tau_factor = - 2 * L.chi * (H*H) * shear_H;
    const Real tau_H_trace = (filter_H.S_xx + filter_H.S_yy + filter_H.S_zz)/3;
    // Residual-stress tensor for filter H : \tau(H) (remove trace)
    const Real tau_H_xx = tau_factor * ( filter_H.S_xx  - tau_H_trace );
    const Real tau_H_xy = tau_factor * ( filter_H.S_xy );
    const Real tau_H_xz = tau_factor * ( filter_H.S_xz );
    const Real tau_H_yy = tau_factor * ( filter_H.S_yy  - tau_H_trace );
    const Real tau_H_yz = tau_factor * ( filter_H.S_yz );
    const Real tau_H_zz = tau_factor * ( filter_H.S_zz  - tau_H_trace );
    const Real dXX = std::pow(L_xx - tau_H_xx + filter_H.tau_xx, 2);
    const Real dXY = std::pow(L_xy - tau_H_xy + filter_H.tau_xy, 2);
    const Real dXZ = std::pow(L_xz - tau_H_xz + filter_H.tau_xz, 2);
    const Real dYY = std::pow(L_yy - tau_H_yy + filter_H.tau_yy, 2);
    const Real dYZ = std::pow(L_yz - tau_H_yz + filter_H.tau_yz, 2);
    const Real dZZ = std::pow(L_zz - tau_H_zz + filter_H.tau_zz, 2);
    // both L and tau have dimension Vel ** 2, L2 error has dim Vel ** 4
    const Real nonDimFac = std::pow(scaleVel, 4);
    return nonDimFac * (dXX + 2*dXY + 2*dXZ + dYY + 2*dYZ + dZZ) / 9;
  }

  KernelGermanoError(const HITstatistics& _stats, std::vector<double>& _locR,
    Real scaleVel) : facVel(scaleVel), stats(_stats), localRewards(_locR) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& i, BlockType& o) const
  {
    const int idx = agentLocInBlock-1, ipx = agentLocInBlock, BID = i.blockID;
    const Real h = i.h_gridpoint;
    localRewards[BID] = 0;
    for (int iz = idx; iz <= ipx; ++iz)
    for (int iy = idx; iy <= ipx; ++iy)
    for (int ix = idx; ix <= ipx; ++ix)
      localRewards[BID] -= germanoError(lab, h, facVel, ix, iy, iz) / 8.0;
  }

  void compute_locR(const BlockInfo & i) const
  {
    const int idx = agentLocInBlock-1, ipx = agentLocInBlock, BID = i.blockID;
    FluidBlock & o = * (FluidBlock *) i.ptrBlock;
    const Real h = i.h_gridpoint;
    localRewards[BID] = 0;
    for (int iz = idx; iz <= ipx; ++iz)
    for (int iy = idx; iy <= ipx; ++iy)
    for (int ix = idx; ix <= ipx; ++ix)
      localRewards[BID] -= germanoError(o, h, facVel, ix, iy, iz) / 8.0;
  }
};

SGS_RL::SGS_RL(SimulationData&s, smarties::Communicator*_comm,
               const int nAgentsPB) : Operator(s), commPtr(_comm),
               nAgentsPerBlock(nAgentsPB)
{
  // TODO relying on chi field does not work is obstacles are present
  // TODO : make sure there are no agents on the same grid point if nAgentsPB>1
  assert(nAgentsPB == 1); // TODO

  std::mt19937& gen = commPtr->getPRNG();
  const std::vector<BlockInfo>& myInfo = sim.vInfo();
  std::uniform_int_distribution<int> disX(0, FluidBlock::sizeX-1);
  std::uniform_int_distribution<int> disY(0, FluidBlock::sizeY-1);
  std::uniform_int_distribution<int> disZ(0, FluidBlock::sizeZ-1);
  agentsIDX.resize(myInfo.size(), -1);
  agentsIDY.resize(myInfo.size(), -1);
  agentsIDZ.resize(myInfo.size(), -1);
  localRewards = std::vector<double>(myInfo.size(), 0);

  for (size_t i=0; i<myInfo.size(); ++i) {
    agentsIDX[i]= disX(gen); agentsIDY[i]= disY(gen); agentsIDZ[i]= disZ(gen);
    //agentsIDX[i] = 8; agentsIDY[i] = 8; agentsIDZ[i] = 8;
  }
}

void SGS_RL::run(const double dt, const bool RLinit, const bool RLover,
                 const HITstatistics& stats, const HITtargetData& target,
                 const Real globalR, const bool bGridAgents)
{
  sim.startProfiler("SGS_RL");
  smarties::Communicator & comm = * commPtr;
  ActionInterpolator actInterp( sim.grid->getResidentBlocksPerDimension(2),
                                sim.grid->getResidentBlocksPerDimension(1),
                                sim.grid->getResidentBlocksPerDimension(0) );

  const Real inpEn = sim.actualInjectionRate, nu = sim.nu, h = sim.uniformH();
  const Real uEps = stats.getKolmogorovU(inpEn, nu);
  const Real lEps = stats.getKolmogorovL(inpEn, nu);
  //const Real tEps = stats.getKolmogorovT(inpEn, nu);
  const Real scaleVel = 1 / std::sqrt(stats.tke); // [T/L]
  const Real scaleGrad = stats.tke / inpEn; // [T]
  const Real scaleLap = scaleGrad * lEps; // [TL]

  // one element per block is a proper agent: will add seq to train data
  // other are nThreads and are only there for thread safety
  // states get overwritten
  //const Real  h_nonDim =  h / stats.getKolmogorovL(stats.dissip_visc, nu);
  //const Real dt_nonDim = dt / stats.getKolmogorovL(stats.dissip_visc, nu);
  const Real tke_nonDim = stats.tke / uEps / uEps;
  const Real visc_nonDim = stats.dissip_visc / inpEn;
  const Real dissi_nonDim = stats.dissip_tot / inpEn;
  //const Real lenInt_nonDim = stats.lambda / stats.l_integral;
  //const Real deltaEn_nonDim = (stats.tke-target.tKinEn) / std::sqrt(inpEn*nu);
  const Real En01_nonDim = stats.E_msr[ 0] / uEps / uEps;
  const Real En02_nonDim = stats.E_msr[ 1] / uEps / uEps;
  const Real En03_nonDim = stats.E_msr[ 2] / uEps / uEps;
  const Real En04_nonDim = stats.E_msr[ 3] / uEps / uEps;
  const Real En05_nonDim = stats.E_msr[ 4] / uEps / uEps;
  const Real En06_nonDim = stats.E_msr[ 5] / uEps / uEps;
  const Real En07_nonDim = stats.E_msr[ 6] / uEps / uEps;
  const Real En08_nonDim = stats.E_msr[ 7] / uEps / uEps;
  const Real En09_nonDim = stats.E_msr[ 8] / uEps / uEps;
  const Real En10_nonDim = stats.E_msr[ 9] / uEps / uEps;
  const Real En11_nonDim = stats.E_msr[10] / uEps / uEps;
  const Real En12_nonDim = stats.E_msr[11] / uEps / uEps;
  const Real En13_nonDim = stats.E_msr[12] / uEps / uEps;
  const Real En14_nonDim = stats.E_msr[13] / uEps / uEps;
  const Real En15_nonDim = stats.E_msr[14] / uEps / uEps;

  const auto getState = [&] (const std::array<Real,13> & locS) {
    return std::vector<double> {
        // locS[ 0],
        locS[ 2], locS[ 3], locS[ 4], locS[ 5], locS[ 6],
        locS[ 7], locS[ 8], locS[ 9], locS[10], locS[11], locS[12],
        // tke_nonDim,
        visc_nonDim, dissi_nonDim,
        // lenInt_nonDim,
        En01_nonDim, En02_nonDim, En03_nonDim, En04_nonDim, En05_nonDim,
        En06_nonDim, En07_nonDim, En08_nonDim, En09_nonDim, En10_nonDim,
        En11_nonDim, En12_nonDim, En13_nonDim, En14_nonDim, En15_nonDim
    };
  };

  using getID_t = size_t(const size_t, const size_t, const int,const int,const int);

  // ONE AGENT PER GRID POINT SETUP:
  // Randomly scattered agent-grid-points that sample the policy for Cs.
  // Rest of grid follows the mean of the policy s.t. grad log pi := 0.
  // Therefore only one element per block is a proper agent: will add EP to
  // train data, while other are nThreads agents and are only there for thread
  // safety: their states get overwritten, actions are policy mean.
  const size_t nBlocks = sim.vInfo().size();
  const std::function<getID_t> getAgentID_grid = [&](const size_t blockID,
    const size_t threadID, const int ix,const int iy,const int iz)
  {
    //const bool bAgent = ix == agentsIDX[blockID] &&
    //                    iy == agentsIDY[blockID] &&
    //                    iz == agentsIDZ[blockID];
    const bool bAgent = ix == agentLocInBlock &&
                        iy == agentLocInBlock &&
                        iz == agentLocInBlock;
    return bAgent? blockID : nBlocks + threadID;
  };

  // ONE AGENT PER FLUID BLOCK SETUP:
  // Agents in block centers and linear interpolate Cs on the grid.
  // The good: (i.) stronger signals for rewards (fewer agents take decisions)
  // (ii.) can use RNN. The bad: Less powerful model, coarse grained state.
  const std::function<getID_t> getAgentID_block = [&](const size_t blockID,
    const size_t threadID, const int ix,const int iy,const int iz)
  {
    return blockID;
  };

  const auto getAgentID = bGridAgents ? getAgentID_grid : getAgentID_block;

  const rlApi_t Finit = [&](const std::array<Real,13> & locS, const size_t bID,
                   const size_t thrID, const int ix, const int iy, const int iz)
  {
    const size_t agentID = getAgentID(bID, thrID, ix, iy, iz);
    comm.sendInitState(getState(locS), agentID);
    return comm.recvAction(agentID)[0];
  };
  const rlApi_t Fcont = [&](const std::array<Real,13> & locS, const size_t bID,
                   const size_t thrID, const int ix, const int iy, const int iz)
  {
    const size_t agentID = getAgentID(bID, thrID, ix, iy, iz);
    //if (agentID==0) printf("locR %e globR %e\n", localRewards[bID], globalR);
    comm.sendState(getState(locS), globalR + localRewards[bID], agentID);
    return comm.recvAction(agentID)[0];
  };
  const rlApi_t Flast = [&](const std::array<Real,13> & locS, const size_t bID,
                   const size_t thrID, const int ix, const int iy, const int iz)
  {
    const size_t agentID = getAgentID(bID, thrID, ix, iy, iz);
    comm.sendLastState(getState(locS), globalR + localRewards[bID], agentID);
    return (Real) 0;
  };
  const rlApi_t sendState = RLinit ? Finit : ( RLover ? Flast : Fcont );

  KernelSGS_RL K_SGS_RL(sendState, actInterp, stats, scaleVel, scaleGrad, scaleLap);

  if(bGridAgents) {
    compute<KernelSGS_RL>(K_SGS_RL);
  } else {
    // new setup : (first get actions for block centers, then interpolate)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vInfo.size(); ++i) K_SGS_RL.state_center(vInfo[i]);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vInfo.size(); ++i) K_SGS_RL.apply_actions(vInfo[i]);
  }

  /*
  KernelGermanoError KlocR(stats, localRewards, scaleVel);
  if(CUP_BLOCK_SIZE > 4) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vInfo.size(); ++i) KlocR.compute_locR(vInfo[i]);
  } else {
    compute<KernelGermanoError>(KlocR);
  }
  */

  sim.stopProfiler();
  check("SGS_RL");
}

int SGS_RL::nStateComponents()
{
  return 28;
}

CubismUP_3D_NAMESPACE_END
