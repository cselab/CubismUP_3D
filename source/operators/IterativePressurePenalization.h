//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_IterativePressurePenalization_h
#define CubismUP_3D_IterativePressurePenalization_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class IterativePressurePenalization : public Operator
{
 protected:
  PoissonSolver * pressureSolver;
  PenalizationGridMPI * penalizationGrid = nullptr;

  void initializeFields();

 public:
  IterativePressurePenalization(SimulationData & s);

  void operator()(const double dt);

  std::string getName() { return "IterativePressurePenalization"; }
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_IterativePressurePenalization_h
