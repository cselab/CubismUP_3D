//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Hugues de Larousslihe.
//

#ifndef CubismUP_3D_SpectralForcing_h
#define CubismUP_3D_SpectralForcing_h

#include "../operators/Operator.h"
#include "Cubism/BlockInfo.h"

CubismUP_3D_NAMESPACE_BEGIN

class SpectralForcing : public Operator
{
  void _cub2fftw() const;
  void _fftw2cub(const Real factor) const;

 public:
  SpectralForcing(SimulationData & s);

  void operator()(const double dt);

  std::string getName() { return "SpectralForcing"; }
};

CubismUP_3D_NAMESPACE_END
#endif //CubismUP_3D_SpectralForcing_h
