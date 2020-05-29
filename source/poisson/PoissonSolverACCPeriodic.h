//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_PoissonSolverPeriodicACC_h
#define CubismUP_3D_PoissonSolverPeriodicACC_h

#include "PoissonSolver.h"

CubismUP_3D_NAMESPACE_BEGIN

class PoissonSolverPeriodic : public PoissonSolver
{
  MPI_Comm c_comm;
  // the local pencil size and the allocation size
  int isize[3], osize[3], istart[3], ostart[3];
  size_t alloc_max;
  //Real * rho_gpu;
  //Real * phi_gpu;
  Real * phi_hat;
  void * plan;

  int cufft_fwd, cufft_bwd;

  const double h = sim.uniformH();
  void solve_multiNode();
  void solve_singleNode();

public:
  PoissonSolverPeriodic(SimulationData & s);

  void solve();
  void testComm();
  ~PoissonSolverPeriodic();
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_PoissonSolverPeriodicACC_h
