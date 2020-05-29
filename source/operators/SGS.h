//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_SGS_h
#define CubismUP_3D_SGS_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class SGS : public Operator
{
  void * _sgsGrid;

public:
  SGS(SimulationData& s);

  ~SGS();

  void operator()(const double dt);

  std::string getName() { return "SGS"; }
};

CubismUP_3D_NAMESPACE_END
#endif
