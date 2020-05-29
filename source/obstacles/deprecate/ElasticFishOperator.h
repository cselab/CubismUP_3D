//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Pantelis Vlachas.
//

#pragma once

#include "obstacles/IF3D_FishOperator.h"

class IF3D_ElasticFishOperator: public IF3D_FishOperator
{
 public:
  IF3D_ElasticFishOperator(SimulationData&s, ArgumentParser&p);
  void writeSDFOnBlocks(const mapBlock2Segs& segmentsPerBlock) override;
  void computeForces(const int stepID, const double time, const double dt, const Real* Uinf, const double NU, const bool bDump) override;
};
