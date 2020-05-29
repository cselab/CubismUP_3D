//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Sid Verma in May 2018.
//

#include "ComputeDissipation.h"
#include "../utils/BufferedLogger.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

class KernelDissipation
{
  public:
  /* Wake power derived by div. theorem applied to pressure and visc forces:
    P_{wake} = 1-\chi \iiint \left(
      - \nabla P \cdot \bm{u}            // Term 1: presPow
      + \mu \bm{u} \cdot \nabla^2\bm{u}  // Term 2: viscPow
      + 2\mu\bm{D} : \bm{D}              // Term 2: viscPow
    \right) dV
  */
  Real circulation[3] = {0,0,0};
  Real linImpulse[3] = {0,0,0};
  Real linMomentum[3] = {0,0,0};
  Real angImpulse[3] = {0,0,0};
  Real angMomentum[3] = {0,0,0};
  Real presPow = 0;
  Real viscPow = 0;
  Real helicity = 0;
  Real kineticEn = 0;
  Real enstrophy = 0;
  //Real maxvorSq = 0;

  const Real dt, nu, center[3];
  const std::array<int, 3> stencil_start = {-1,-1,-1}, stencil_end = {2, 2, 2};
  const StencilInfo stencil{-1,-1,-1, 2,2,2, false, {FE_U,FE_V,FE_W,FE_P}};

  KernelDissipation(double _dt, const Real ext[3], Real _nu)
  : dt(_dt), nu(_nu), center{ext[0]/2, ext[1]/2, ext[2]/2} { }

  template <typename Lab, typename BlockType>
  void operator()(Lab & lab, const BlockInfo& info, BlockType& o)
  {
    const Real h = info.h_gridpoint;
    const Real hCube = std::pow(h,3), inv2h = .5 / h, invHh = 1/(h*h);

    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix)
    {
      const FluidElement &L =lab(ix,iy,iz);
      const FluidElement &LW=lab(ix-1,iy,iz), &LE=lab(ix+1,iy,iz);
      const FluidElement &LS=lab(ix,iy-1,iz), &LN=lab(ix,iy+1,iz);
      const FluidElement &LF=lab(ix,iy,iz-1), &LB=lab(ix,iy,iz+1);
      const Real X = L.chi;
      Real p[3]; info.pos(p, ix, iy, iz);
      const Real PX = p[0]-center[0], PY = p[1]-center[1], PZ = p[2]-center[2];
      // vorticity
      const Real WX = inv2h * ( (LN.w-LS.w) - (LB.v-LF.v) );
      const Real WY = inv2h * ( (LB.u-LF.u) - (LE.w-LW.w) );
      const Real WZ = inv2h * ( (LE.v-LW.v) - (LN.u-LS.u) );
      //  - \nabla P \cdot \bm{u}
      const Real dPdx = inv2h * (LE.p - LW.p);
      const Real dPdy = inv2h * (LN.p - LS.p);
      const Real dPdz = inv2h * (LB.p - LF.p);
      presPow -= (1-X) * hCube * ( dPdx * L.u + dPdy * L.v + dPdz * L.w);
      //  + \mu \bm{u} \cdot \nabla^2 \bm{u}
      const Real lapU = invHh * ( LE.u+LW.u+LN.u+LS.u+LB.u+LF.u -6*L.u );
      const Real lapV = invHh * ( LE.v+LW.v+LN.v+LS.v+LB.v+LF.v -6*L.v );
      const Real lapW = invHh * ( LE.w+LW.w+LN.w+LS.w+LB.w+LF.w -6*L.w );
      const Real V1 = lapU * L.u + lapV * L.v + lapW * L.w;
      // + 2 \mu \bm{D} : \bm{D}
      const Real D11 = inv2h*(LE.u - LW.u); // shear stresses
      const Real D22 = inv2h*(LN.v - LS.v); // shear stresses
      const Real D33 = inv2h*(LB.w - LF.w); // shear stresses
      const Real D12 = inv2h*(LN.u - LS.u + LE.v - LW.v)/2; // shear stresses
      const Real D13 = inv2h*(LB.u - LF.u + LE.w - LW.w)/2; // shear stresses
      const Real D23 = inv2h*(LN.w - LS.w + LB.v - LF.v)/2; // shear stresses
      const Real V2 = D11*D11 +D22*D22 +D33*D33 +2*(D12*D12 +D13*D13 +D23*D23);
      viscPow += (1-X) * hCube*nu * ( V1 + 2*V2 );

      // three linear invariants (conserved in inviscid and viscous flows)
      // conservation of vorticity: int w dx = 0
      circulation[0] += hCube * WX;
      circulation[1] += hCube * WY;
      circulation[2] += hCube * WZ;
      // conservation of linear impulse: int u dx = 0.5 int (x cross w) dx
      linImpulse[0] += hCube/2 * ( PY * WZ - PZ * WY );
      linImpulse[1] += hCube/2 * ( PZ * WX - PX * WZ );
      linImpulse[2] += hCube/2 * ( PX * WY - PY * WX );
      linMomentum[0] += hCube * L.u;
      linMomentum[1] += hCube * L.v;
      linMomentum[2] += hCube * L.w;
      //conserve ang imp.: int (x cross u)dx = 1/3 int (x cross (x cross w) )dx
      // = 1/3 int x (w \cdot x) - w (x \cdot x) ) dx (some terms cancel)
      angImpulse[0] += hCube/3 * ( PX*(PY*WY + PZ*WZ) - WX*(PY*PY + PZ*PZ) );
      angImpulse[1] += hCube/3 * ( PY*(PX*WX + PZ*WZ) - WY*(PX*PX + PZ*PZ) );
      angImpulse[2] += hCube/3 * ( PZ*(PX*WX + PY*WY) - WZ*(PX*PX + PY*PY) );
      angMomentum[0] += hCube * ( PY * L.w - PZ * L.v );
      angMomentum[1] += hCube * ( PZ * L.u - PX * L.w );
      angMomentum[2] += hCube * ( PX * L.v - PY * L.u );
      // two quadratic invariants: kinetic energy (from solver) and helicity (conserved in inviscid flows)
      helicity += hCube * ( WX * L.u + WY * L.v + WZ * L.w );
      kineticEn += hCube * ( L.u*L.u + L.v*L.v + L.w*L.w )/2;
      // two more important: maxvor and enstrophy
      const Real omegasq = std::sqrt(WX * WX + WY * WY + WZ * WZ);
      enstrophy += omegasq;
      //maxVorSq = std::max(maxVorSq, omegasq);
    }
  }
};
}

void ComputeDissipation::operator()(const double dt)
{
  if(sim.freqDiagnostics == 0 || sim.step % sim.freqDiagnostics) return;

  sim.startProfiler("Dissip Kernel");
  const int nthreads = omp_get_max_threads();
  std::vector<KernelDissipation*> diss(nthreads, nullptr);
  #pragma omp parallel for schedule(static, 1)
  for(int i=0; i<nthreads; ++i)
    diss[i] = new KernelDissipation(dt, sim.extent.data(), sim.nu);

  compute<KernelDissipation>(diss);
  sim.stopProfiler();

  double RDX[20] = { 0.0 };
  sim.startProfiler("Dissip Reduce");
  for(int i=0; i<nthreads; i++) {
    RDX[ 0] += diss[i]->circulation[0];
    RDX[ 1] += diss[i]->circulation[1];
    RDX[ 2] += diss[i]->circulation[2];
    RDX[ 3] += diss[i]->linImpulse[0];
    RDX[ 4] += diss[i]->linImpulse[1];
    RDX[ 5] += diss[i]->linImpulse[2];
    RDX[ 6] += diss[i]->linMomentum[0];
    RDX[ 7] += diss[i]->linMomentum[1];
    RDX[ 8] += diss[i]->linMomentum[2];
    RDX[ 9] += diss[i]->angImpulse[0];
    RDX[10] += diss[i]->angImpulse[1];
    RDX[11] += diss[i]->angImpulse[2];
    RDX[12] += diss[i]->angMomentum[0];
    RDX[13] += diss[i]->angMomentum[1];
    RDX[14] += diss[i]->angMomentum[2];
    RDX[15] += diss[i]->presPow;
    RDX[16] += diss[i]->viscPow;
    RDX[17] += diss[i]->helicity;
    RDX[18] += diss[i]->kineticEn;
    RDX[19] += diss[i]->enstrophy;
    delete diss[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, RDX, 20,MPI_DOUBLE, MPI_SUM,grid->getCartComm());

  if(sim.rank==0 and not sim.muteAll) {
    std::stringstream &fileDissip = logger.get_stream("diagnostics.dat");
    if(sim.step==0)
     fileDissip<<"step_id time circ_x circ_y circ_y linImp_x linImp_y linImp_z "
     "linMom_x linMom_y linMom_z angImp_x angImp_y angImp_z angMom_x angMom_y "
     "angMom_z presPow viscPow helicity kineticEn enstrophy"<<std::endl;

    fileDissip<<sim.step<<" "<<sim.time<<" "<<
     RDX[ 0]<<" "<<RDX[ 1]<<" "<<RDX[ 2]<<" "<<RDX[ 3]<<" "<<RDX[ 4]<<" "<<
     RDX[ 5]<<" "<<RDX[ 6]<<" "<<RDX[ 7]<<" "<<RDX[ 8]<<" "<<RDX[ 9]<<" "<<
     RDX[10]<<" "<<RDX[11]<<" "<<RDX[12]<<" "<<RDX[13]<<" "<<RDX[14]<<" "<<
     RDX[15]<<" "<<RDX[16]<<" "<<RDX[17]<<" "<<RDX[18]<<" "<<RDX[19]<<std::endl;
  }
  sim.stopProfiler();

  check("ComputeDissipation");
}

CubismUP_3D_NAMESPACE_END
