//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and
//  Hugues de Laroussilhe (huguesdelaroussilhe@gmail.com).
//

#ifndef CubismUP_3D_SpectralManipFFTW_h
#define CubismUP_3D_SpectralManipFFTW_h

#include "SpectralManip.h"

CubismUP_3D_NAMESPACE_BEGIN

class SpectralManipFFTW : public SpectralManip
{
  const size_t nz_hat = gsize[2]/2+1;
  ptrdiff_t alloc_local=0;
  ptrdiff_t local_n0=0, local_0_start=0;
  ptrdiff_t local_n1=0, local_1_start=0;
  void * fwd_u = nullptr;
  void * fwd_v = nullptr;
  void * fwd_w = nullptr;
  #ifdef ENERGY_FLUX_SPECTRUM
  void * fwd_j = nullptr;
  #endif
  void * bwd_u = nullptr;
  void * bwd_v = nullptr;
  void * bwd_w = nullptr;

public:

  SpectralManipFFTW(SimulationData & s);
  ~SpectralManipFFTW() override;

  void prepareFwd() override;
  void prepareBwd() override;

  void runFwd() const override;
  void runBwd() const override;

  void _compute_forcing() override;
  void _compute_IC(const std::vector<Real> &K,
                   const std::vector<Real> &E) override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_SpectralManipFFTW_h
