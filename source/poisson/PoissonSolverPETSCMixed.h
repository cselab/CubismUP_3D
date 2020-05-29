//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//


#ifndef CubismUP_3D_PoissonSolverMixed_PETSC_h
#define CubismUP_3D_PoissonSolverMixed_PETSC_h
#ifdef CUP_PETSC

#include "PoissonSolver.h"

CubismUP_3D_NAMESPACE_BEGIN

class PoissonSolverMixed_PETSC : public PoissonSolver
{
  const std::string solver;
  Real pLast = 0;
  int peidx_0() const {  int ret[3]; grid.peindex(ret); return ret[0]; }
  int peidx_1() const {  int ret[3]; grid.peindex(ret); return ret[1]; }
  int peidx_2() const {  int ret[3]; grid.peindex(ret); return ret[2]; }
  const int peidx[3] = {peidx_0(), peidx_1(), peidx_2()};
  int ilower[3] = {
    (int) myN[0] * peidx[0],
    (int) myN[1] * peidx[1],
    (int) myN[2] * peidx[2]
  };
  int iupper[3] = {
    (int) myN[0] * (peidx[0]+1) - 1,
    (int) myN[1] * (peidx[1]+1) - 1,
    (int) myN[2] * (peidx[2]+1) - 1
  };
  int iGridEnd[3] = {(int)gsize[0]-1, (int)gsize[1]-1, (int)gsize[2]-1};
  const size_t fixed_idx  = linaccess(myN[0]-2, myN[1]-2, myN[2]-2);

  const size_t fixed_m1x = linaccess(myN[0]-3, myN[1]-2, myN[2]-2);
  const size_t fixed_m1y = linaccess(myN[0]-2, myN[1]-3, myN[2]-2);
  const size_t fixed_m1z = linaccess(myN[0]-2, myN[1]-2, myN[2]-3);

  const size_t fixed_p1x = linaccess(myN[0]-1, myN[1]-2, myN[2]-2);
  const size_t fixed_p1y = linaccess(myN[0]-2, myN[1]-1, myN[2]-2);
  const size_t fixed_p1z = linaccess(myN[0]-2, myN[1]-2, myN[2]-1);

  const bool bRankHoldsFixedDOF =
    iupper[0]==iGridEnd[0] && iupper[1]==iGridEnd[1] && iupper[2]==iGridEnd[2];

  inline size_t linaccess(const size_t ix,const size_t iy,const size_t iz) const
  {
    return ix + myN[0] * iy + myN[0]*myN[1] * iz;
  }

 public:

  struct PetscData;
  PetscData * S;

  void solve() override;

  PoissonSolverMixed_PETSC(SimulationData& s);

  std::string getName() {
    return "petsc";
  }

  ~PoissonSolverMixed_PETSC();
};

CubismUP_3D_NAMESPACE_END
#endif // CUP_PETSC
#endif // CubismUP_3D_PoissonSolverMixed_PETSC_h
