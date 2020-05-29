//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "FluidSolidForces.h"
#include "../obstacles/ObstacleVector.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

struct KernelComputeForces : public ObstacleVisitor
{
  ObstacleVector * const obstacle_vector;
  const double nu, dt;
  LabMPI * lab_ptr = nullptr;
  const BlockInfo * info_ptr = nullptr;

  const int stencil_start[3] = {-1, -1, -1}, stencil_end[3] = {2, 2, 2};
  StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W}};

  KernelComputeForces(double _nu, double _dt, ObstacleVector* ov) :
    obstacle_vector(ov), nu(_nu), dt(_dt) { }

  void operator()(LabMPI&lab,const BlockInfo&info,const FluidBlock&block)
  {
    // first store the lab and info, then do visitor
    lab_ptr = & lab;
    info_ptr = & info;
    ObstacleVisitor* const base = static_cast<ObstacleVisitor*> (this);
    assert( base not_eq nullptr );
    obstacle_vector->Accept( base );
    lab_ptr = nullptr;
    info_ptr = nullptr;
  }

  void visit(Obstacle* const op)
  {
    LabMPI& l = * lab_ptr;
    const BlockInfo& info = * info_ptr;
    assert(lab_ptr not_eq nullptr && info_ptr not_eq nullptr);
    const std::vector<ObstacleBlock*>& obstblocks = op->getObstacleBlocks();
    ObstacleBlock*const o = obstblocks[info.blockID];
    if (o == nullptr) return;
    if (o->nPoints == 0) return;
    assert(o->filled);

    const std::array<double,3> CM = op->getCenterOfMass();
    const std::array<double,3> omega = op->getAngularVelocity();
    const std::array<double,3> uTrans = op->getTranslationVelocity();
    double velUnit[3] = {0., 0., 0.};
    const double vel_norm = std::sqrt(uTrans[0]*uTrans[0]
                                    + uTrans[1]*uTrans[1]
                                    + uTrans[2]*uTrans[2]);
    if (vel_norm>1e-9) {
        velUnit[0] = uTrans[0] / vel_norm;
        velUnit[1] = uTrans[1] / vel_norm;
        velUnit[2] = uTrans[2] / vel_norm;
    }

    const double _1oH = nu / info.h_gridpoint;
    //const Real _h3 = std::pow(info.h_gridpoint,3);

    //loop over elements of block info that have nonzero gradChi
    for(int i=0; i<o->nPoints; i++)
    {
      Real p[3];
      const int ix = o->surface[i]->ix;
      const int iy = o->surface[i]->iy;
      const int iz = o->surface[i]->iz;

      // WAS A SOURCE OF HUGE BUG - due to unzeroed values in surfData arrays. The kid had forgotten to initialize allocated arrays in surfData to zero!!
      //if(o->chi[iz][iy][ix] < 1e-16) continue;

      info.pos(p, ix, iy, iz);

      //shear stresses
      const double D11 =    _1oH*(l(ix+1,iy,iz).u - l(ix-1,iy,iz).u);
      const double D22 =    _1oH*(l(ix,iy+1,iz).v - l(ix,iy-1,iz).v);
      const double D33 =    _1oH*(l(ix,iy,iz+1).w - l(ix,iy,iz-1).w);
      const double D12 = .5*_1oH*(l(ix,iy+1,iz).u - l(ix,iy-1,iz).u
                                 +l(ix+1,iy,iz).v - l(ix-1,iy,iz).v);
      const double D13 = .5*_1oH*(l(ix,iy,iz+1).u - l(ix,iy,iz-1).u
                                 +l(ix+1,iy,iz).w - l(ix-1,iy,iz).w);
      const double D23 = .5*_1oH*(l(ix,iy+1,iz).w - l(ix,iy-1,iz).w
                                 +l(ix,iy,iz+1).v - l(ix,iy,iz-1).v);

      //normals computed with Towers 2009
      // Actually using the volume integral, since (\iint -P \hat{n} dS) = (\iiint -\nabla P dV). Also, P*\nabla\Chi = \nabla P
      // penalty-accel and surf-force match up if resolution is high enough (200 points per fish)
      const double P = l(ix,iy,iz).p;
      const double normX = o->surface[i]->dchidx; //*h^3 (multiplied in dchidx)
      const double normY = o->surface[i]->dchidy; //*h^3 (multiplied in dchidy)
      const double normZ = o->surface[i]->dchidz; //*h^3 (multiplied in dchidz)
      const double fXV = D11 * normX + D12 * normY + D13 * normZ;
      const double fYV = D12 * normX + D22 * normY + D23 * normZ;
      const double fZV = D13 * normX + D23 * normY + D33 * normZ;
      const double fXP = -P * normX, fYP = -P * normY, fZP = -P * normZ;
      const double fXT = fXV+fXP, fYT = fYV+fYP, fZT = fZV+fZP;

      //store:
      o->ss[i] = o->sectionMarker[iz][iy][ix];
      o->pX[i] = p[0]; o->pY[i] = p[1]; o->pZ[i] = p[2];
      o->P[i] = P; o->fX[i] = fXT; o->fY[i] = fYT; o->fZ[i] = fZT;
      o->vxDef[i] = o->udef[iz][iy][ix][0]; o->vX[i] = l(ix,iy,iz).u;
      o->vyDef[i] = o->udef[iz][iy][ix][1]; o->vY[i] = l(ix,iy,iz).v;
      o->vzDef[i] = o->udef[iz][iy][ix][2]; o->vZ[i] = l(ix,iy,iz).w;

      //additive quantities:
      // o->surface += o->surface[i]->delta;
      o->gammax += normY*o->vZ[i] - normZ*o->vY[i];
      o->gammay += normZ*o->vX[i] - normX*o->vZ[i];
      o->gammaz += normX*o->vY[i] - normY*o->vX[i];
      //forces (total, visc, pressure):
      o->forcex   += fXT; o->forcey   += fYT; o->forcez   += fZT;
      o->forcex_V += fXV; o->forcey_V += fYV; o->forcez_V += fZV;
      o->forcex_P += fXP; o->forcey_P += fYP; o->forcez_P += fZP;
      //torque:
      o->torquex  += (p[1]-CM[1])*fZT - (p[2]-CM[2])*fYT;
      o->torquey  += (p[2]-CM[2])*fXT - (p[0]-CM[0])*fZT;
      o->torquez  += (p[0]-CM[0])*fYT - (p[1]-CM[1])*fXT;
      /*
      if(tempIt->second->sectionMarker[iz][iy][ix] > 0){
        const double * const pHinge2 = tempIt->second->hinge2LabFrame;
        (*measures)[19] += (p[1]-pHinge2[1])*fZT - (p[2]-pHinge2[2])*fYT;
        (*measures)[20] += (p[2]-pHinge2[2])*fXT - (p[0]-pHinge2[0])*fZT;
        (*measures)[21] += (p[0]-pHinge2[0])*fYT - (p[1]-pHinge2[1])*fXT;
      }
      */
      //thrust, drag:
      const double forcePar= fXT*velUnit[0] +fYT*velUnit[1] +fZT*velUnit[2];
      o->thrust += .5*(forcePar + std::fabs(forcePar));
      o->drag   -= .5*(forcePar - std::fabs(forcePar));

      //power output (and negative definite variant which ensures no elastic energy absorption)
      // This is total power, for overcoming not only deformation, but also the oncoming velocity. Work done by fluid, not by the object (for that, just take -ve)
      const double powOut = fXT * o->vX[i] + fYT * o->vY[i] + fZT * o->vZ[i];
      //deformation power output (and negative definite variant which ensures no elastic energy absorption)
      const double powDef = fXT*o->vxDef[i] + fYT*o->vyDef[i] + fZT*o->vzDef[i];
      o->Pout        += powOut; o->PoutBnd     += std::min((double)0, powOut);
      o->defPower    += powDef; o->defPowerBnd += std::min((double)0, powDef);

      // Compute P_locomotion = Force*(uTrans + uRot)
      const double rVec[3] = {p[0]-CM[0], p[1]-CM[1], p[2]-CM[2]};
      const double uSolid[3] = {
	        uTrans[0] + omega[1]*rVec[2] - rVec[1]*omega[2],
	        uTrans[1] + omega[2]*rVec[0] - rVec[2]*omega[0],
	        uTrans[2] + omega[0]*rVec[1] - rVec[0]*omega[1]
      };
      o->pLocom += fXT*uSolid[0] + fYT*uSolid[1] + fZT*uSolid[2];
    }
  }
};

struct DumpWake
{
  double t;
  const int stencil_start[3] = {-1, -1, -1}, stencil_end[3] = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {{FE_P}}};
  const Real *Uinf;
  const double *CM, length, theta = 0.15;
  FILE* const pFile;


  DumpWake(const Real*u, const double*cm, FILE*pf, const double l):
    t(0), Uinf(u), CM(cm), length(l), pFile(pf) { }

  template <typename Lab, typename BlockType>
  void operator()(Lab& lab, const BlockInfo& info, BlockType& b)
  {
    const double _1oH = .5 / info.h_gridpoint;
    const double h = info.h_gridpoint;
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
      Real p[3];
      info.pos(p, ix, iy, iz);
      p[0] -= CM[0];
      p[1] -= CM[1];
      p[2] -= CM[2];
      if (std::fabs(p[2]) > 0.5*h) continue;

      //const double x = p[0]*std::cos(theta) + p[1]*std::sin(theta);
      //const double y = p[1]*std::cos(theta) - p[0]*std::sin(theta);
      //if (x<0.50*length || x>3.00*length) continue; //behind swimmer
      //if (y<-.35*length || y>0.35*length) continue;
      if (p[1]<.0 || p[1]>.2 || p[0]<0 || p[0]>.6) continue;
      const double gradPx = _1oH*(lab(ix+1,iy,iz).p-lab(ix-1,iy,iz).p);
      const double gradPy = _1oH*(lab(ix,iy+1,iz).p-lab(ix,iy-1,iz).p);
      const double d[6] = {
        p[0], p[1], b(ix,iy,iz).u+Uinf[0], b(ix,iy,iz).v+Uinf[1], gradPx, gradPy
      };
      fwrite(d,sizeof(double),6,pFile);
    }
  }
};

}

void ComputeForces::operator()(const double dt)
{
  if(sim.obstacle_vector->nObstacles() == 0) return;

  sim.startProfiler("Obst. Forces");
  const int nthreads = omp_get_max_threads();
  std::vector<KernelComputeForces*> K(nthreads, nullptr);
  #pragma omp parallel for schedule(static,1)
  for(int i=0; i<nthreads; ++i)
    K[i] = new KernelComputeForces(sim.nu, sim.dt, sim.obstacle_vector);

  compute<KernelComputeForces>(K);

  for(int i=0; i<nthreads; i++) delete K[i];
  // do the final reductions and so on
  sim.obstacle_vector->computeForces();
  sim.stopProfiler();
  check("ComputeForces");
}

CubismUP_3D_NAMESPACE_END
