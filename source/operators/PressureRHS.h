//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_PressureRHS_h
#define CubismUP_3D_PressureRHS_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN
#define PENAL_THEN_PRES

class PressureRHS : public Operator
{
  PenalizationGridMPI * penalizationGrid = nullptr;
 public:
  PressureRHS(SimulationData & s);
  ~PressureRHS();

  void operator()(const double dt);

  std::string getName() { return "PressureRHS"; }
};

CubismUP_3D_NAMESPACE_END
#endif
