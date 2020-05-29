//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#ifndef CubismUP_3D_StefanFish_h
#define CubismUP_3D_StefanFish_h

#include "Fish.h"

CubismUP_3D_NAMESPACE_BEGIN

class StefanFish: public Fish
{
protected:
  Real origC[2] = {(Real)0, (Real)0};
  Real origAng = 0;
public:
  StefanFish(SimulationData&s, cubism::ArgumentParser&p);
  void save(std::string filename = std::string()) override;
  void restart(std::string filename) override;
  void create() override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_StefanFish_h
