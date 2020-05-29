//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Penalization_h
#define CubismUP_3D_Penalization_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class Penalization : public Operator
{
 public:
  Penalization(SimulationData & s);

  void operator()(const double dt);

  std::string getName() { return "Penalization"; }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Penalization_h
