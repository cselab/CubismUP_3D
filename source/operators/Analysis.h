//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Analysis_h
#define CubismUP_3D_Analysis_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class SpectralAnalysis;

class Analysis : public Operator
{
  SpectralAnalysis * sA = nullptr;
public:
  Analysis(SimulationData& s);

  ~Analysis();

  void operator()(const double dt);

  std::string getName() { return "Analysis"; }
};

CubismUP_3D_NAMESPACE_END
#endif
