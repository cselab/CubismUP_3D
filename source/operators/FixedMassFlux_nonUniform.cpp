//
//  CubismUP_3D
//
//  Written by Jacopo Canton ( jcanton@ethz.ch ).
//  Copyright (c) 2017 ETHZ. All rights reserved.
//

#include "FixedMassFlux_nonUniform.h"

CubismUP_3D_NAMESPACE_BEGIN using namespace cubism;

static Real avgUx_nonUniform(const std::vector<BlockInfo>& myInfo,
                             const Real* const uInf, const Real volume)
{
  // Average Ux on the simulation volume :
  //   Sum on the xz-plane (uniform)
  //   Integral along Y    (non-uniform)
  //
  // <Ux>_{xz} (iy) = 1/(Nx.Ny) . \Sum_{ix, iz} u(ix,iy,iz)
  //
  // <Ux>_{xyz} = 1/Ly . \Sum_{iy} <Ux>_{xz} (iy) . h_y(*,iy,*)
  //            = /1(Nx.Ny.Ly) . \Sum_{ix,iy,iz} u(ix,iy,iz).h_y(ix,iy,iz)
  Real avgUx = 0.;
  const int nBlocks = myInfo.size();

  #pragma omp parallel for schedule(static) reduction(+ : avgUx)
  for (int i = 0; i < nBlocks; i++) {
    const BlockInfo& info = myInfo[i];
    const FluidBlock& b = *(const FluidBlock*)info.ptrBlock;

    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      Real h[3]; info.spacing(h, ix, iy, iz);
      avgUx += (b(ix, iy, iz).u + uInf[0]) * h[0] * h[1] * h[2];
    }
  }
  avgUx = avgUx / volume;
  return avgUx;
}

class KernelFixedMassFlux_nonUniform
{
  const Real dt, scale, y_max;

 public:
  KernelFixedMassFlux_nonUniform(double _dt, double _scale, double _y_max)
      : dt(_dt), scale(_scale), y_max(_y_max) { }

  void operator()(const BlockInfo& info, FluidBlock& o) const
  {
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy) {
      Real p[3]; info.pos(p, 0, iy, 0);
      const Real y = p[1];
      for (int ix = 0; ix < FluidBlock::sizeX; ++ix)
          o(ix, iy, iz).u += 6 * scale * y/y_max * (1.0 - y/y_max);
    }
  }
};

FixedMassFlux_nonUniform::FixedMassFlux_nonUniform(SimulationData& s)
    : Operator(s) {}

void FixedMassFlux_nonUniform::operator()(const double dt)
{
  sim.startProfiler("FixedMassFlux");

  // fix base_u_avg and y_max AD HOC for channel flow
  Real u_avg_msr, delta_u;
  const Real volume = sim.extent[0]*sim.extent[1]*sim.extent[2];
  const Real y_max = sim.extent[1];
  const Real u_avg = 2.0/3.0 * sim.uMax_forced;

  u_avg_msr = avgUx_nonUniform(vInfo, sim.uinf.data(), volume);
  MPI_Allreduce(MPI_IN_PLACE, &u_avg_msr, 1, MPI_DOUBLE, MPI_SUM,
                grid->getCartComm());

  delta_u = u_avg - u_avg_msr;
  const Real reTau = std::sqrt(std::fabs(delta_u/sim.dt)) / sim.nu;

  const Real scale = 6*delta_u;
  if (sim.rank == 0) {
    printf(
        "Measured <Ux>_V = %25.16e,\n"
        "target   <Ux>_V = %25.16e,\n"
        "delta    <Ux>_V = %25.16e,\n"
        "scale           = %25.16e,\n"
        "Re_tau          = %25.16e,\n",
        u_avg_msr, u_avg, delta_u, scale, reTau);
  }
  KernelFixedMassFlux_nonUniform K(sim.dt, scale, y_max);
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<vInfo.size(); i++)
    K(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);

  sim.stopProfiler();
  check("FixedMassFlux");
}

CubismUP_3D_NAMESPACE_END
