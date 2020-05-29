//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_InitialConditions_h
#define CubismUP_3D_InitialConditions_h

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

class InitialConditions : public Operator
{
 public:
  InitialConditions(SimulationData & s) : Operator(s) { }

  template<typename K>
  inline void run(const K kernel) {
    #pragma omp parallel for schedule(static)
    for (size_t i=0; i<vInfo.size(); i++)
      kernel(vInfo[i], *(FluidBlock*)vInfo[i].ptrBlock);
  }

  void operator()(const double dt);

  std::string getName() { return "IC"; }
};

CubismUP_3D_NAMESPACE_END
#endif
