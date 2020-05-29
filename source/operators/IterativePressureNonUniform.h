//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_IterativePressureNonUniform_h
#define CubismUP_3D_IterativePressureNonUniform_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class IterativePressureNonUniform : public Operator
{
 protected:
  PoissonSolver * pressureSolver;

 public:
  IterativePressureNonUniform(SimulationData & s);

  void operator()(const double dt);

  std::string getName() { return "IterativePressureNonUniform"; }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_IterativePressureNonUniform_h
