//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Sid Verma in May 2018.
//

#ifndef CubismUP_3D_ComputeDissipation_h
#define CubismUP_3D_ComputeDissipation_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class ComputeDissipation : public Operator
{
  Real oldKE=0.0;
public:
  ComputeDissipation(SimulationData & s) : Operator(s) { }
  void operator()(const double dt);
  std::string getName() { return "Dissipation"; }
};

CubismUP_3D_NAMESPACE_END
#endif
