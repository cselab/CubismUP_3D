//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Naca_h
#define CubismUP_3D_Naca_h

#include "Fish.h"

CubismUP_3D_NAMESPACE_BEGIN

class Naca: public Fish
{
  double Apitch, Fpitch, Ppitch, Mpitch, Fheave, Aheave;

 public:
  Naca(SimulationData&s, cubism::ArgumentParser&p);
  void update() override;
  void computeVelocities() override;
  using intersect_t = std::vector<std::vector<VolumeSegment_OBB*>>;
  void writeSDFOnBlocks(const intersect_t& segmentsPerBlock) override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Naca_h
