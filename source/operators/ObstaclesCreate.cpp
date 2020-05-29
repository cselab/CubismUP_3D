//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "ObstaclesCreate.h"
#include "../obstacles/ObstacleVector.h"
#include "../utils/MatArrayMath.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

using CHIMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE];
using UDEFMAT = Real[CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][CUP_BLOCK_SIZE][3];
static constexpr Real EPS = std::numeric_limits<Real>::epsilon();

class KernelCharacteristicFunction
{
  using v_v_ob = std::vector<std::vector<ObstacleBlock*>*>;
  const v_v_ob & vec_obstacleBlocks;

  public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_TMPU}}};
  /* // now not in use
  static Real computeCHI(const Real h, const Real dist,
                         const Real distPx, const Real distMx,
                         const Real distPy, const Real distMy,
                         const Real distPz, const Real distMz)
  {
    if (dist > +2*h || dist < -2*h) return dist > 0 ? 1 : 0;

    Real gradUX = distPx-distMx, gradUY = distPy-distMy, gradUZ = distPz-distMz;
    const Real IplusX = distPx<0? 0 : distPx, IminuX = distMx<0? 0 : distMx;
    const Real IplusY = distPy<0? 0 : distPy, IminuY = distMy<0? 0 : distMy;
    const Real IplusZ = distPz<0? 0 : distPz, IminuZ = distMz<0? 0 : distMz;
    // gradI: first primitive of H(x): I(x) = int_0^x H(y) dy
    Real gradIX = IplusX-IminuX, gradIY = IplusY-IminuY, gradIZ = IplusZ-IminuZ;
    const Real gradUSq = gradUX*gradUX + gradUY*gradUY + gradUZ*gradUZ;
    return (gradIX*gradUX +gradIY*gradUY +gradIZ*gradUZ)/std::max(gradUSq, EPS);
  }
  */

  KernelCharacteristicFunction(const v_v_ob& v) : vec_obstacleBlocks(v) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& b) const
  {
    const Real h = info.h_gridpoint, inv2h = .5/h, fac1 = .5*h*h, vol = h*h*h;

    for (size_t obst_id = 0; obst_id<vec_obstacleBlocks.size(); obst_id++)
    {
      const auto& obstacleBlocks = * vec_obstacleBlocks[obst_id];
      ObstacleBlock* const o = obstacleBlocks[info.blockID];
      if(o == nullptr) continue;
      CHIMAT & __restrict__ CHI = o->chi;
      const CHIMAT & __restrict__ SDF = o->sdf;
      o->CoM_x = 0; o->CoM_y = 0; o->CoM_z = 0; o->mass  = 0;

      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        // here I read fist from SDF to deal with obstacles sharing block
        if (SDF[iz][iy][ix] > +2*h || SDF[iz][iy][ix] < -2*h)
        {
          CHI[iz][iy][ix] = SDF[iz][iy][ix] > 0 ? 1 : 0;
        }
        else
        {
          const Real distPx =lab(ix+1,iy,iz).tmpU, distMx =lab(ix-1,iy,iz).tmpU;
          const Real distPy =lab(ix,iy+1,iz).tmpU, distMy =lab(ix,iy-1,iz).tmpU;
          const Real distPz =lab(ix,iy,iz+1).tmpU, distMz =lab(ix,iy,iz-1).tmpU;
          // gradU
          const Real gradUX = inv2h*(distPx - distMx);
          const Real gradUY = inv2h*(distPy - distMy);
          const Real gradUZ = inv2h*(distPz - distMz);
          const Real gradUSq = gradUX*gradUX+gradUY*gradUY+gradUZ*gradUZ + EPS;

          const Real IplusX = distPx < 0 ? 0 : distPx;
          const Real IminuX = distMx < 0 ? 0 : distMx;
          const Real IplusY = distPy < 0 ? 0 : distPy;
          const Real IminuY = distMy < 0 ? 0 : distMy;
          const Real IplusZ = distPz < 0 ? 0 : distPz;
          const Real IminuZ = distMz < 0 ? 0 : distMz;
          // gradI: first primitive of H(x): I(x) = int_0^x H(y) dy
          const Real gradIX = inv2h*(IplusX - IminuX);
          const Real gradIY = inv2h*(IplusY - IminuY);
          const Real gradIZ = inv2h*(IplusZ - IminuZ);
          const Real numH = gradIX*gradUX + gradIY*gradUY + gradIZ*gradUZ;
          CHI[iz][iy][ix]  =        numH/gradUSq;
        }

        Real p[3]; info.pos(p, ix,iy,iz);
        b(ix,iy,iz).chi = std::max(CHI[iz][iy][ix], b(ix,iy,iz).chi);
        o->CoM_x += CHI[iz][iy][ix] * vol * p[0];
        o->CoM_y += CHI[iz][iy][ix] * vol * p[1];
        o->CoM_z += CHI[iz][iy][ix] * vol * p[2];
        o->mass  += CHI[iz][iy][ix] * vol;

        static constexpr Real surfdh = 0; // SURFDH
        // allows shifting the SDF outside the body:
        const Real sdf = SDF[iz][iy][ix] + h*surfdh; // negative outside
        if (sdf > +2*h || sdf < -2*h) continue; // no need to compute gradChi

        const Real distPx =lab(ix+1,iy,iz).tmpU + h*surfdh;
        const Real distMx =lab(ix-1,iy,iz).tmpU + h*surfdh;
        const Real distPy =lab(ix,iy+1,iz).tmpU + h*surfdh;
        const Real distMy =lab(ix,iy-1,iz).tmpU + h*surfdh;
        const Real distPz =lab(ix,iy,iz+1).tmpU + h*surfdh;
        const Real distMz =lab(ix,iy,iz-1).tmpU + h*surfdh;
        const Real gradUX = inv2h*(distPx - distMx);
        const Real gradUY = inv2h*(distPy - distMy);
        const Real gradUZ = inv2h*(distPz - distMz);
        const Real gradUSq = gradUX*gradUX+gradUY*gradUY+gradUZ*gradUZ + EPS;
        const Real HplusX = std::fabs(distPx)<EPS ? 0.5 : (distPx<0? 0 : 1);
        const Real HminuX = std::fabs(distMx)<EPS ? 0.5 : (distMx<0? 0 : 1);
        const Real HplusY = std::fabs(distPy)<EPS ? 0.5 : (distPy<0? 0 : 1);
        const Real HminuY = std::fabs(distMy)<EPS ? 0.5 : (distMy<0? 0 : 1);
        const Real HplusZ = std::fabs(distPz)<EPS ? 0.5 : (distPz<0? 0 : 1);
        const Real HminuZ = std::fabs(distMz)<EPS ? 0.5 : (distMz<0? 0 : 1);
        const Real gradHX = (HplusX - HminuX);
        const Real gradHY = (HplusY - HminuY);
        const Real gradHZ = (HplusZ - HminuZ);
        const Real numD = gradHX*gradUX + gradHY*gradUY + gradHZ*gradUZ;
        const Real Delta = fac1 * numD/gradUSq; //h^3 * Delta
        if (Delta>EPS) o->write(ix, iy, iz, Delta, gradUX, gradUY, gradUZ);
      }

      o->allocate_surface();
    }
  }
};

class KernelCharacteristicFunction_nonUniform
{
  using v_v_ob = std::vector<std::vector<ObstacleBlock*>*>;
  const v_v_ob & vec_obstacleBlocks;

  public:
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_TMPU}}};

  KernelCharacteristicFunction_nonUniform(const v_v_ob& v) : vec_obstacleBlocks(v) {}

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& b) const
  {
    const BlkCoeffX &cx =b.fd_cx.first, &cy =b.fd_cy.first, &cz =b.fd_cz.first;

    for (size_t obst_id = 0; obst_id<vec_obstacleBlocks.size(); obst_id++)
    {
      const auto& obstacleBlocks = * vec_obstacleBlocks[obst_id];
      ObstacleBlock*const o = obstacleBlocks[info.blockID];
      if(o == nullptr) continue;
      CHIMAT & __restrict__ CHI = o->chi;
      const CHIMAT & __restrict__ SDF = o->sdf;
      o->CoM_x = 0; o->CoM_y = 0; o->CoM_z = 0; o->mass  = 0;

      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix)
      {
        Real p[3]; info.pos(p, ix,iy,iz);
        Real h[3]; info.spacing(h, ix, iy, iz);
        const Real vol = h[0]*h[1]*h[2];
        const Real maxH = std::max({h[0],h[1],h[2]});
        // towers would need two points on each side, but
        // I do not really know what to do here, let's be safe:
        if (SDF[iz][iy][ix] > +3*maxH || SDF[iz][iy][ix] < -3*maxH)
        {
          CHI[iz][iy][ix] = SDF[iz][iy][ix] > 0 ? 1 : 0;
        }
        else
        {
          const Real dist   =lab(ix,iy,iz).tmpU;
          const Real distPx =lab(ix+1,iy,iz).tmpU, distMx =lab(ix-1,iy,iz).tmpU;
          const Real distPy =lab(ix,iy+1,iz).tmpU, distMy =lab(ix,iy-1,iz).tmpU;
          const Real distPz =lab(ix,iy,iz+1).tmpU, distMz =lab(ix,iy,iz-1).tmpU;
          // gradU
          const Real gradUX = __FD_2ND(ix, cx, distMx, dist, distPx);
          const Real gradUY = __FD_2ND(iy, cy, distMy, dist, distPy);
          const Real gradUZ = __FD_2ND(iz, cz, distMz, dist, distPz);
          const Real gradUSq = gradUX*gradUX+gradUY*gradUY+gradUZ*gradUZ + EPS;

          const Real IplusX = distPx < 0 ? 0 : distPx;
          const Real IminuX = distMx < 0 ? 0 : distMx;
          const Real IplusY = distPy < 0 ? 0 : distPy;
          const Real IminuY = distMy < 0 ? 0 : distMy;
          const Real IplusZ = distPz < 0 ? 0 : distPz;
          const Real IminuZ = distMz < 0 ? 0 : distMz;
          const Real Icent  = dist   < 0 ? 0 : dist;
          // gradI: first primitive of H(x): I(x) = int_0^x H(y) dy
          const Real gradIX = __FD_2ND(ix, cx, IminuX, Icent, IplusX);
          const Real gradIY = __FD_2ND(iy, cy, IminuY, Icent, IplusY);
          const Real gradIZ = __FD_2ND(iz, cz, IminuZ, Icent, IplusZ);
          const Real numH = gradIX*gradUX + gradIY*gradUY + gradIZ*gradUZ;
          CHI[iz][iy][ix]  =        numH/gradUSq;

          const double HplusX = std::fabs(distPx)<EPS ? 0.5 : (distPx<0? 0 : 1);
          const double HminuX = std::fabs(distMx)<EPS ? 0.5 : (distMx<0? 0 : 1);
          const double HplusY = std::fabs(distPy)<EPS ? 0.5 : (distPy<0? 0 : 1);
          const double HminuY = std::fabs(distMy)<EPS ? 0.5 : (distMy<0? 0 : 1);
          const double HplusZ = std::fabs(distPz)<EPS ? 0.5 : (distPz<0? 0 : 1);
          const double HminuZ = std::fabs(distMz)<EPS ? 0.5 : (distMz<0? 0 : 1);
          const double Hcent  = std::fabs(dist  )<EPS ? 0.5 : (dist  <0? 0 : 1);
          const double gradHX = __FD_2ND(ix, cx, HminuX, Hcent, HplusX);
          const double gradHY = __FD_2ND(iy, cy, HminuY, Hcent, HplusY);
          const double gradHZ = __FD_2ND(iz, cz, HminuZ, Hcent, HplusZ);
          const double numD = gradHX*gradUX + gradHY*gradUY + gradHZ*gradUZ;
          const double Delta = vol * numD/gradUSq; //h^3 * Delta
          if (Delta>EPS) o->write(ix, iy, iz, Delta, gradUX, gradUY, gradUZ);
        }

        b(ix,iy,iz).chi = std::max(CHI[iz][iy][ix], b(ix,iy,iz).chi);
        o->CoM_x += vol * CHI[iz][iy][ix] * p[0];
        o->CoM_y += vol * CHI[iz][iy][ix] * p[1];
        o->CoM_z += vol * CHI[iz][iy][ix] * p[2];
        o->mass  += vol * CHI[iz][iy][ix];
      }
      o->allocate_surface();
    }
  }
};

struct KernelComputeGridCoM : public ObstacleVisitor
{
  FluidGridMPI * const grid;
  const std::vector<cubism::BlockInfo>& vInfo = grid->getBlocksInfo();

  KernelComputeGridCoM(FluidGridMPI*g) : grid(g) { }

  void visit(Obstacle* const obstacle)
  {
    double com[4] = {0.0, 0.0, 0.0, 0.0};
    const auto& obstblocks = obstacle->getObstacleBlocks();
    #pragma omp parallel for schedule(static,1) reduction(+ : com[:4])
    for (size_t i=0; i<obstblocks.size(); i++) {
      if(obstblocks[i] == nullptr) continue;
      com[0] += obstblocks[i]->mass;
      com[1] += obstblocks[i]->CoM_x;
      com[2] += obstblocks[i]->CoM_y;
      com[3] += obstblocks[i]->CoM_z;
    }
    MPI_Allreduce(MPI_IN_PLACE, com, 4,MPI_DOUBLE,MPI_SUM, grid->getCartComm());

    assert(com[0]>std::numeric_limits<Real>::epsilon());
    obstacle->centerOfMass[0] = com[1]/com[0];
    obstacle->centerOfMass[1] = com[2]/com[0];
    obstacle->centerOfMass[2] = com[3]/com[0];
  }
};

struct KernelIntegrateUdefMomenta : public ObstacleVisitor
{
  ObstacleVector * const obstacle_vector;
  const cubism::BlockInfo * info_ptr = nullptr;
  inline double dvol(const cubism::BlockInfo&info, const int x, const int y, const int z) const {
    double h[3]; info.spacing(h, x, y, z);
    return h[0] * h[1] * h[2];
  }

  KernelIntegrateUdefMomenta(ObstacleVector* ov) : obstacle_vector(ov) {}

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

  void visit(Obstacle* const obstacle)
  {
    const BlockInfo& info = * info_ptr;
    assert(info_ptr not_eq nullptr);
    const auto& obstblocks = obstacle->getObstacleBlocks();
    ObstacleBlock*const o = obstblocks[info.blockID];
    if (o == nullptr) return;

    const std::array<double,3> CM = obstacle->getCenterOfMass();
    //We use last momentum computed by this method to stabilize the computation
    //of the ang vel. This is because J is really tiny.
    const std::array<double,3> oldCorrVel = {{
      obstacle->transVel_correction[0],
      obstacle->transVel_correction[1],
      obstacle->transVel_correction[2]
    }};

    const CHIMAT & __restrict__ CHI = o->chi;
    const UDEFMAT & __restrict__ UDEF = o->udef;
    double &VV = o->V;
    double &FX = o->FX, &FY = o->FY, &FZ = o->FZ;
    double &TX = o->TX, &TY = o->TY, &TZ = o->TZ;
    double &J0 = o->J0, &J1 = o->J1, &J2 = o->J2;
    double &J3 = o->J3, &J4 = o->J4, &J5 = o->J5;
    VV = 0; FX = 0; FY = 0; FZ = 0; TX = 0; TY = 0; TZ = 0;
    J0 = 0; J1 = 0; J2 = 0; J3 = 0; J4 = 0; J5 = 0;

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      if (CHI[iz][iy][ix] <= 0) continue;
      double p[3]; info.pos(p, ix, iy, iz);
      const double dv = dvol(info, ix, iy, iz), X = CHI[iz][iy][ix];
      p[0] -= CM[0];
      p[1] -= CM[1];
      p[2] -= CM[2];
      const double dUs = UDEF[iz][iy][ix][0] - oldCorrVel[0];
      const double dVs = UDEF[iz][iy][ix][1] - oldCorrVel[1];
      const double dWs = UDEF[iz][iy][ix][2] - oldCorrVel[2];
      VV += X * dv;
      FX += X * UDEF[iz][iy][ix][0] * dv;
      FY += X * UDEF[iz][iy][ix][1] * dv;
      FZ += X * UDEF[iz][iy][ix][2] * dv;
      TX += X * ( p[1]*dWs - p[2]*dVs ) * dv;
      TY += X * ( p[2]*dUs - p[0]*dWs ) * dv;
      TZ += X * ( p[0]*dVs - p[1]*dUs ) * dv;
      J0 += X * ( p[1]*p[1]+p[2]*p[2] ) * dv; J3 -= X * p[0]*p[1] * dv;
      J1 += X * ( p[0]*p[0]+p[2]*p[2] ) * dv; J4 -= X * p[0]*p[2] * dv;
      J2 += X * ( p[0]*p[0]+p[1]*p[1] ) * dv; J5 -= X * p[1]*p[2] * dv;
    }
  }
};

struct KernelAccumulateUdefMomenta : public ObstacleVisitor
{
  FluidGridMPI * const grid;
  const std::vector<cubism::BlockInfo>& vInfo = grid->getBlocksInfo();
  const bool justDebug = false;

  KernelAccumulateUdefMomenta(FluidGridMPI*g, bool dbg = false) :
    grid(g), justDebug(dbg) {}

  void visit(Obstacle* const obst)
  {
    double M[13] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    const auto& oBlock = obst->getObstacleBlocks();
    #pragma omp parallel for schedule(static,1) reduction(+ : M[:13])
    for (size_t i=0; i<oBlock.size(); i++) {
      if(oBlock[i] == nullptr) continue;
      M[ 0] += oBlock[i]->V ;
      M[ 1] += oBlock[i]->FX; M[ 2] += oBlock[i]->FY; M[ 3] += oBlock[i]->FZ;
      M[ 4] += oBlock[i]->TX; M[ 5] += oBlock[i]->TY; M[ 6] += oBlock[i]->TZ;
      M[ 7] += oBlock[i]->J0; M[ 8] += oBlock[i]->J1; M[ 9] += oBlock[i]->J2;
      M[10] += oBlock[i]->J3; M[11] += oBlock[i]->J4; M[12] += oBlock[i]->J5;
    }
    const auto comm = grid->getCartComm();
    MPI_Allreduce(MPI_IN_PLACE, M, 13, MPI_DOUBLE, MPI_SUM, comm);
    assert(M[0] > EPS);

    const GenV AM = {{ M[ 4], M[ 5], M[ 6] }};
    const SymM J =  {{ M[ 7], M[ 8], M[ 9], M[10], M[11], M[12] }};
    const SymM invJ = invertSym(J);

    if(justDebug) {
      assert(std::fabs(M[ 1])<100*EPS);
      assert(std::fabs(M[ 2])<100*EPS);
      assert(std::fabs(M[ 3])<100*EPS);
      assert(std::fabs(AM[0])<100*EPS);
      assert(std::fabs(AM[1])<100*EPS);
      assert(std::fabs(AM[2])<100*EPS);
    } else {
      //solve avel = invJ \dot angMomentum
      obst->mass                   = M[ 0];
      obst->transVel_correction[0] = M[ 1] / M[0];
      obst->transVel_correction[1] = M[ 2] / M[0];
      obst->transVel_correction[2] = M[ 3] / M[0];
      obst->J[0] = M[ 7]; obst->J[1] = M[ 8]; obst->J[2] = M[ 9];
      obst->J[3] = M[10]; obst->J[4] = M[11]; obst->J[5] = M[12];
      obst->angVel_correction[0] = invJ[0]*AM[0] +invJ[3]*AM[1] +invJ[4]*AM[2];
      obst->angVel_correction[1] = invJ[3]*AM[0] +invJ[1]*AM[1] +invJ[5]*AM[2];
      obst->angVel_correction[2] = invJ[4]*AM[0] +invJ[5]*AM[1] +invJ[2]*AM[2];
    }
  }
};

struct KernelRemoveUdefMomenta : public ObstacleVisitor
{
  FluidGridMPI * const grid;
  const std::vector<cubism::BlockInfo>& vInfo = grid->getBlocksInfo();

  KernelRemoveUdefMomenta(FluidGridMPI*g) : grid(g) { }

  void visit(Obstacle* const obstacle)
  {
    const std::array<double,3> angVel_correction = {{
      obstacle->angVel_correction[0],
      obstacle->angVel_correction[1],
      obstacle->angVel_correction[2]
    }};
    const std::array<double,3> transVel_correction = {{
      obstacle->transVel_correction[0],
      obstacle->transVel_correction[1],
      obstacle->transVel_correction[2]
    }};

    #ifdef CUP_VERBOSE
     if(sim.rank==0)
        printf("Obstacle %d moment corrections lin:[%f %f %f] ang:[%f %f %f]\n",
          obstacle->obstacleID, transVel_correction[0], transVel_correction[1],
          transVel_correction[2], angVel_correction[0], angVel_correction[1],
          angVel_correction[2]);
    #endif

    const std::array<double,3> CM = obstacle->getCenterOfMass();
    const auto & obstacleBlocks = obstacle->getObstacleBlocks();

    #pragma omp parallel for schedule(dynamic, 1)
    for(size_t i=0; i < vInfo.size(); i++)
    {
      const BlockInfo& info = vInfo[i];
      const auto pos = obstacleBlocks[info.blockID];
      if(pos == nullptr) continue;
      UDEFMAT & __restrict__ UDEF = pos->udef;
      for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
      for(int iy=0; iy<FluidBlock::sizeY; ++iy)
      for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        double p[3]; info.pos(p, ix, iy, iz);
        p[0] -= CM[0]; p[1] -= CM[1]; p[2] -= CM[2];
        const double rotVel_correction[3] = {
          angVel_correction[1]*p[2] - angVel_correction[2]*p[1],
          angVel_correction[2]*p[0] - angVel_correction[0]*p[2],
          angVel_correction[0]*p[1] - angVel_correction[1]*p[0]
        };
        UDEF[iz][iy][ix][0] -= transVel_correction[0] + rotVel_correction[0];
        UDEF[iz][iy][ix][1] -= transVel_correction[1] + rotVel_correction[1];
        UDEF[iz][iy][ix][2] -= transVel_correction[2] + rotVel_correction[2];
      }
    }
  }
};

}

void CreateObstacles::operator()(const double dt)
{
  if(sim.obstacle_vector->nObstacles() == 0) return;

  sim.startProfiler("Obst Reset");
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < vInfo.size(); ++i)
  {
    FluidBlock& b = *(FluidBlock*)vInfo[i].ptrBlock;
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      b(ix,iy,iz).chi = 0; // will be accessed by max with pos def qtity
      b(ix,iy,iz).tmpU = -1; // will be accessed by max with pos/neg qtity
    }
  }
  sim.stopProfiler();

  // Obstacles' advection must be done after we perform penalization:
  sim.uinf = sim.obstacle_vector->updateUinf();
  sim.obstacle_vector->update();

  sim.startProfiler("Obst SDF");
  { // put signed distance function on the grid
    sim.obstacle_vector->create();
  }
  sim.stopProfiler();

  sim.startProfiler("Obst CHI");
  if(sim.bUseStretchedGrid)
  {
    auto vecOB = sim.obstacle_vector->getAllObstacleBlocks();
    const KernelCharacteristicFunction_nonUniform K(vecOB);
    compute<KernelCharacteristicFunction_nonUniform>(K);
  }
  else
  {
    auto vecOB = sim.obstacle_vector->getAllObstacleBlocks();
    const KernelCharacteristicFunction K(vecOB);
    compute<KernelCharacteristicFunction>(K);
  }
  sim.stopProfiler();

  sim.startProfiler("Obst CoM");
  { // compute actual CoM given the CHI on the grid
    ObstacleVisitor* visitor = new KernelComputeGridCoM(grid);
    sim.obstacle_vector->Accept(visitor);
    delete visitor;
  }
  sim.stopProfiler();

  sim.startProfiler("Obst Int Mom");
  { // integrate momenta by looping over grid
    #pragma omp parallel
    { // each thread needs to call its own non-const operator() function
      KernelIntegrateUdefMomenta K(sim.obstacle_vector);
      #pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
    }
  }
  sim.stopProfiler();

  sim.startProfiler("Obst Rdx Mom");
  { // reduce momenta across blocks and MPI
    ObstacleVisitor* visitor = new KernelAccumulateUdefMomenta(grid);
    sim.obstacle_vector->Accept(visitor);
    delete visitor;
  }
  sim.stopProfiler();

  sim.startProfiler("Obst 0 Mom");
  { // remove momenta from udef
    ObstacleVisitor* visitor = new KernelRemoveUdefMomenta(grid);
    sim.obstacle_vector->Accept(visitor);
    delete visitor;
  }
  sim.stopProfiler();

  #ifndef NDEBUG
  { // integrate momenta by looping over grid
    #pragma omp parallel
    { // each thread needs to call its own non-const operator() function
      KernelIntegrateUdefMomenta K(sim.obstacle_vector);
      #pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < vInfo.size(); ++i) K(vInfo[i]);
    }
    ObstacleVisitor* visitor = new KernelAccumulateUdefMomenta(grid, true);
    sim.obstacle_vector->Accept(visitor);
    delete visitor;
  }
  #endif

  sim.startProfiler("Obst finalize");
  sim.obstacle_vector->finalize(); // whatever else the obstacle needs
  sim.stopProfiler();
  check("CreateObstacles");
}

CubismUP_3D_NAMESPACE_END
