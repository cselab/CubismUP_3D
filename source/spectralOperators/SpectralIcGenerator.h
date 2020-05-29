//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Hugues de Laroussilhe.
//

#ifndef CubismUP_3D_SpectralIcGenerator_h
#define CubismUP_3D_SpectralIcGenerator_h

#include "../SimulationData.h"
#include "Cubism/BlockInfo.h"

#include <vector>
#include <cassert>
#include <cstring>

CubismUP_3D_NAMESPACE_BEGIN

class SpectralManip;

class SpectralIcGenerator
{
  SimulationData & sim;
public:
  typedef typename FluidGridMPI::BlockType BlockType;

  SpectralIcGenerator(SimulationData &s);
  ~SpectralIcGenerator() {}

  void run();

private:

  void _generateTarget(std::vector<Real>&, std::vector<Real>&, SpectralManip&);
  void _fftw2cub(const SpectralManip&) const;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_SpectralIcGenerator_h
