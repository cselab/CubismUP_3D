//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Christian Conti.
//

#include "ExternalForcing.h"

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace {

template<int DIRECTION>
class KernelExternalForcing
{
  const double gradPdT;
 public:
  KernelExternalForcing(double _gradPxdT) : gradPdT(_gradPxdT) { }
  void operator()(const BlockInfo& info, FluidBlock& b) const
  {
    for(int iz=0; iz<FluidBlock::sizeZ; ++iz)
    for(int iy=0; iy<FluidBlock::sizeY; ++iy)
    for(int ix=0; ix<FluidBlock::sizeX; ++ix) {
        if (DIRECTION == 0) b(ix,iy,iz).u    += gradPdT;
        if (DIRECTION == 1) b(ix,iy,iz).v    += gradPdT;
        if (DIRECTION == 2) b(ix,iy,iz).w    += gradPdT;
    }
  }
};
}

void ExternalForcing::operator()(const double dt)
{
  sim.startProfiler("Forcing Kernel");
  const int dir = sim.BCy_flag==wall ? 1 : 2;
  const Real H = sim.extent[dir];
  const Real gradPdt = 8*sim.uMax_forced*sim.nu/H/H * dt;

  const KernelExternalForcing<0> kernel( gradPdt );
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<vInfo.size(); i++)
      kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
  sim.stopProfiler();
  check("ExternalForcing");
}

CubismUP_3D_NAMESPACE_END
