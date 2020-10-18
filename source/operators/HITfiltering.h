//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_HITfiltering_h
#define CubismUP_3D_HITfiltering_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN


class HITfiltering : public Operator
{
public:
  HITfiltering(SimulationData& s);

  void operator()(const double dt);

  std::string getName() { return "HITfiltering"; }
};

class StructureFunctions : public Operator
{
  std::mt19937 gen;
  const Real computeInterval = sim.timeAnalysis / 10;
  Real nextComputeTime = 0;

  std::array<double, 6> pick_ref_point();

public:
  StructureFunctions(SimulationData& s);

  void operator()(const double dt);

  std::string getName() { return "StructureFunctions"; }
};

CubismUP_3D_NAMESPACE_END
#endif
