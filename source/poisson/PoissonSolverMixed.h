//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_PoissonSolverMixed_h
#define CubismUP_3D_PoissonSolverMixed_h

#include "PoissonSolver.h"

CubismUP_3D_NAMESPACE_BEGIN

class PoissonSolverMixed : public PoissonSolver
{
  void * fwd, * bwd;
  ptrdiff_t alloc_local=0,local_n0=0,local_0_start=0,local_n1=0,local_1_start=0;
  const double h = sim.uniformH();
  inline bool DFT_X() const { return sim.BCx_flag == periodic; }
  inline bool DFT_Y() const { return sim.BCy_flag == periodic; }
  inline bool DFT_Z() const { return sim.BCz_flag == periodic; }

 protected:

  template<bool DFTX, bool DFTY, bool DFTZ> void _solve()
  {
    // if BC flag == 1 fourier, else cosine transform
    const Real normX = (DFTX ? 1.0 : 0.5) / gsize[0];
    const Real normY = (DFTY ? 1.0 : 0.5) / gsize[1];
    const Real normZ = (DFTZ ? 1.0 : 0.5) / gsize[2];
    const Real waveFacX = (DFTX ? 2 : 1) * M_PI / gsize[0];
    const Real waveFacY = (DFTY ? 2 : 1) * M_PI / gsize[1];
    const Real waveFacZ = (DFTZ ? 2 : 1) * M_PI / gsize[2];
    // factor 1/h here is becz input to this solver is h^3 * RHS:
    // (other h^2 goes away from FD coef or wavenumeber coef)
    const Real norm_factor = (normX / h) * normY * normZ;
    Real *const in_out = data;
    const long nKx = static_cast<long>(gsize[0]);
    const long nKy = static_cast<long>(gsize[1]);
    const long nKz = static_cast<long>(gsize[2]);
    const long shifty = static_cast<long>(local_1_start);

    // BALANCE TWO PROBLEMS:
    // - if only grid consistent odd DOF and even DOF do not 'talk' to each others
    // - if only spectral then nont really div free
    // COMPROMISE: define a tolerance that balances two effects
    //static constexpr Real tol = 0.01;
    //static constexpr Real tol = 1;

    #pragma omp parallel for schedule(static)
    for(long lj = 0; lj<static_cast<long>(local_n1); ++lj)
    {
      const long j = shifty + lj; //memory index plus shift due to decomp
      const long ky = DFTY ? ((j <= nKy/2) ? j : nKy-j) : j;
      const Real rky2 = std::pow( (ky + (DFTY? 0 : (Real)0.5)) * waveFacY, 2);
      //const Real denY = (1-tol) * (std::cos(2*waveFacY*j)-1)/2 - tol*rky2;
      const Real denY = - rky2;

      for(long  i = 0;  i<static_cast<long>(gsize[0]); ++ i)
      {
        const long kx = DFTX ? ((i <= nKx/2) ? i : nKx-i) : i;
        const Real rkx2 = std::pow( (kx + (DFTX? 0 : (Real)0.5)) * waveFacX, 2);
        //const Real denX = (1-tol) * (std::cos(2*waveFacX*i)-1)/2 - tol*rkx2;
        const Real denX = - rkx2;

        for(long  k = 0;  k<static_cast<long>(gsize[2]); ++ k)
        {
          const size_t linidx = (lj*gsize[0] +i)*gsize[2] + k;
          const long kz = DFTZ ? ((k <= nKz/2) ? k : nKz-k) : k;
          const Real rkz2 = std::pow( (kz + (DFTZ? 0 : (Real)0.5)) * waveFacZ, 2);
          //const Real denZ = (1-tol) * (std::cos(2*waveFacZ*k)-1)/2 - tol*rkz2;
          const Real denZ = - rkz2;

          in_out[linidx] *= norm_factor/(denX + denY + denZ);
        }
      }

    }

    //if (shifty==0 && DFTX && DFTY && DFTZ) in_out[0] = 0;
    if (shifty==0) in_out[0] = 0;
  }

 public:

  PoissonSolverMixed(SimulationData & s);

  void solve();

  ~PoissonSolverMixed();
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_PoissonSolverMixed_h
