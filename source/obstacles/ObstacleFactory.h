//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#ifndef CubismUP_3D_ObstacleFactory_h
#define CubismUP_3D_ObstacleFactory_h

#include "ObstacleVector.h"

namespace cubism { class ArgumentParser; }

CubismUP_3D_NAMESPACE_BEGIN

struct ObstacleAndExternalArguments;  // For ExternalObstacle.

class ObstacleFactory
{
  SimulationData & sim;
public:
  ObstacleFactory(SimulationData & s) : sim(s) { }

  /* Add obstacles defined with `-factory` and `-factory-content` arguments. */
  void addObstacles(cubism::ArgumentParser &parser);

  /* Add obstacles specified with a given string. */
  void addObstacles(const std::string &factoryContent);

  /* Helper function for external codes to avoid std::make_shared on their side... */
  void addObstacle(const ObstacleAndExternalArguments &args);
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_ObstacleFactory_h
