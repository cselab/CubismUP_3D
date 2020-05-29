//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_CarlingFish_h
#define CubismUP_3D_CarlingFish_h

#include "Fish.h"

CubismUP_3D_NAMESPACE_BEGIN

class CarlingFishMidlineData;

class CarlingFish: public Fish
{
  CarlingFishMidlineData* readHingeParams(cubism::ArgumentParser&p);
  CarlingFishMidlineData* readBurstCoastParams(cubism::ArgumentParser&p);
 public:
  CarlingFish(SimulationData&s, cubism::ArgumentParser&p);

  #ifdef RL_LAYER
    void execute(const int i,const double t,const vector<double>a) override;
  #endif
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_CarlingFish_h
