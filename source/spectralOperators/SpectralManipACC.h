//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_SpectralManipACCFFT_h
#define CubismUP_3D_SpectralManipACCFFT_h

#include "SpectralManip.h"

CubismUP_3D_NAMESPACE_BEGIN

struct myCUDAstreams;

class SpectralManipACC : public SpectralManip
{
  MPI_Comm acc_comm;
  // the local pencil size and the allocation size
  int isize[3], osize[3], istart[3], ostart[3];
  size_t alloc_max;
  Real * gpu_u;
  Real * gpu_v;
  Real * gpu_w;
  void * plan;
  int cufft_fwd, cufft_bwd;

  myCUDAstreams * const streams;

public:

  SpectralManipACC(SimulationData & s);
  ~SpectralManipACC() override;

  void prepareFwd() override;
  void prepareBwd() override;

  void runFwd() const override;
  void runBwd() const override;

  void _compute_forcing() override;
  void _compute_IC(const std::vector<Real> &K,
                   const std::vector<Real> &E) override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_SpectralManipACCFFT_h
