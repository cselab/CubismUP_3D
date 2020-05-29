//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_ComputeForces_h
#define CubismUP_3D_ComputeForces_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class ComputeForces : public Operator
{
 public:
  ComputeForces(SimulationData & s) : Operator(s) {}

  void operator()(const double dt);

  std::string getName() { return "ComputeForces"; }
};

CubismUP_3D_NAMESPACE_END
#endif
