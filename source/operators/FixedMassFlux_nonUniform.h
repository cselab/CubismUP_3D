//
//  CubismUP_3D
//
//  Written by Jacopo Canton ( jcanton@ethz.ch ).
//  Copyright (c) 2017 ETHZ. All rights reserved.
//

#ifndef CubismUP_3D_FixedMassFlux_nonUniform_h
#define CubismUP_3D_FixedMassFlux_nonUniform_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class FixedMassFlux_nonUniform : public Operator
{
public:
  FixedMassFlux_nonUniform(SimulationData &s);

  void operator()(const double dt);

  std::string getName(){ return "FixedMassFlux_nonUniform"; }
};

CubismUP_3D_NAMESPACE_END
#endif
