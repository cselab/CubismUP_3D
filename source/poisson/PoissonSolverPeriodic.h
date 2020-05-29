//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_PoissonSolverPeriodic_h
#define CubismUP_3D_PoissonSolverPeriodic_h

#include "PoissonSolver.h"

CubismUP_3D_NAMESPACE_BEGIN

class PoissonSolverPeriodic : public PoissonSolver
{
  void * fwd, * bwd;
  const size_t nz_hat = gsize[2]/2+1;
  const double h = sim.uniformH();
  ptrdiff_t alloc_local=0, local_n0=0, local_0_start=0, local_n1=0, local_1_start=0;

 protected:

  void _solve();

 public:

  PoissonSolverPeriodic(SimulationData & s);

  void solve();

  ~PoissonSolverPeriodic();
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_PoissonSolverPeriodic_h
