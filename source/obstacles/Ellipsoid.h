//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Ellipsoid_h
#define CubismUP_3D_Ellipsoid_h

#include "Obstacle.h"

CubismUP_3D_NAMESPACE_BEGIN

class Ellipsoid: public Obstacle
{
  const double radius;
  Real e0, e1, e2;
  //special case: startup with unif accel to umax in tmax, and then decel to 0
  bool accel_decel = false;
  double umax = 0, tmax = 1;

public:

  Ellipsoid(SimulationData&s, cubism::ArgumentParser&p);

  void create() override;
  void finalize() override;
  void computeVelocities() override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Ellipsoid_h
