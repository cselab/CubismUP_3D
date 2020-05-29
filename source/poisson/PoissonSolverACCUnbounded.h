//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_PoissonSolverUnboundedACC_h
#define CubismUP_3D_PoissonSolverUnboundedACC_h

#include "PoissonSolver.h"

CubismUP_3D_NAMESPACE_BEGIN

class PoissonSolverUnbounded : public PoissonSolver
{
  // the local pencil size and the allocation size
  int isize[3], osize[3], istart[3], ostart[3];
  const int mx = 2*gsize[0]-1, my = 2*gsize[1]-1, mz = 2*gsize[2]-1;
  const size_t mz_pad = mz/2 +1, myftNx = (mx+1)/m_size;
  const int szFft[3] = {(int) myftNx, (int) gsize[1], (int) gsize[2] };
  const int szCup[3] = {std::min(szFft[0],(int)myN[0]),(int)myN[1],(int)myN[2]};
  const double h = sim.uniformH();

  MPI_Comm sort_comm, c_comm;
  int s_rank;
  size_t alloc_max;
  Real * fft_rhs;
  Real * gpuGhat;
  Real * gpu_rhs;
  void * plan;
  MPI_Datatype submat;

  inline int map2accfftRank(const int _rank, const int peidx[3]) const {
    if (myftNx == myN[0]) { // we want x to be the fast index
      return peidx[0] +sim.nprocsx*(peidx[1] +sim.nprocsy*peidx[2]);
    } else {
      return _rank/2 + (m_size/2) * (_rank % 2);
    }
  }
public:
  PoissonSolverUnbounded(SimulationData & s);

  void solve();

  void cub2padded() const;
  void padded2cub() const;
  void padded2gpu() const;
  void gpu2padded() const;
  ~PoissonSolverUnbounded();
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_PoissonSolverUnboundedACC_h
