//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and
//  Hugues de Laroussilhe (huguesdelaroussilhe@gmail.com).
//

#ifndef CubismUP_3D_HITstatistics_h
#define CubismUP_3D_HITstatistics_h

#include <vector>
#include <cassert>
#include <cstring> // memset

CubismUP_3D_NAMESPACE_BEGIN

struct HITstatistics
{
  HITstatistics(const int maxGridSize, const Real maxBoxLength):
    N(maxGridSize), L(maxBoxLength),
    k_msr(new Real[nBin]), E_msr(new Real[nBin]), Eflux(new Real[nBin])
  {
    //printf("maxGridSize %d %d %d\n", maxGridSize, N, nyquist);
    reset();
    for (int i = 0; i<nBin; ++i) k_msr[i] = (i+1) * 2*M_PI / L;
  }

  HITstatistics(const HITstatistics&c) : N(c.N), L(c.L),
    k_msr(new Real[nBin]), E_msr(new Real[nBin]), Eflux(new Real[nBin])
  {
    //printf("maxGridSize %d %d %d\n", c.N, N, nyquist);
    reset();
    for (int i = 0; i<nBin; ++i) k_msr[i] = (i+1) * 2*M_PI / L;
  }

  ~HITstatistics()
  {
    delete [] k_msr;
    delete [] E_msr;
    delete [] Eflux;
  }

  void reset()
  {
    tke = 0; tke_filtered = 0; tke_prev = 0;
    dissip_tot = 0; dissip_visc = 0;
    tau_integral = 0; l_integral = 0;
    lambda = 0; uprime = 0; Re_lambda = 0;
    memset(E_msr, 0, nBin * sizeof(Real));
    memset(Eflux, 0, nBin * sizeof(Real));
  }

  void updateDerivedQuantities(const Real _nu, const Real _dt,
                               const Real injectionRate = -1.0)
  {
    dt = _dt;
    nu = _nu;
    if(injectionRate >= 0) {
      dissip_tot = (tke_prev - tke) / dt;
      // tke_prev for next step will be current plus whatever we inject:
      tke_prev = tke + dt * injectionRate;
    }
    uprime = std::sqrt(2.0/3.0 * tke);
    lambda = std::sqrt(15 * nu / dissip_visc) * uprime;
    Re_lambda = uprime * lambda / nu;
    tau_integral = l_integral / uprime;
    //tau_integral = l_integral * M_PI/(2*pow3(uprime));
  }

  Real getSimpleSpectrumFit(const Real _k, const Real _eps) const
  {
    return 5.7 * std::pow(_eps, 2/3.0) * std::pow(_k, -5/3.0);
  }

  void getTargetSpectrumFit(const Real _eps, const Real _nu,
                            std::vector<Real>& K, std::vector<Real>& E) const
  {
    assert(K.size() == E.size() && K.size() > 0);
    // const Real gradFit = 0.8879967 std::sqrt(_eps) / std::sqrt(_nu);
    const Real LintFit = getIntegralLengthFit(_eps, _nu);
    const Real Lkolmogorov = getKolmogorovL(_eps, _nu);
    const Real C  = 5.7, CI = 1e-3, CE = 0;
    const Real BE = 4.9; // 5.2 from theory
    const Real P0 = 600; // should be 2, but we force large scales

    for (size_t i=0; i<K.size(); ++i) {
      K[i] = (i+0.5) * 2 * M_PI / L;
      const Real KI = K[i] * LintFit, KE4 = std::pow(K[i] * Lkolmogorov,4);
      const Real FL = std::pow(KI / (KI + CI), 5/3.0 + P0 );
      const Real FE = std::exp(-BE*(std::pow(KE4 + std::pow(CE,4), 0.25 ) -CE));
      E[i] = C * std::pow(_eps, 2/3.0) * std::pow(K[i], -5/3.0) * FL * FE;
    }
  }

  static Real getIntegralTimeFit(const Real _eps, const Real _nu)
  {
    return 0.93931475 * std::pow(_eps, -1/3.0) * std::pow(_nu, 1/6.0);
  }
  static Real getIntegralLengthFit(const Real _eps, const Real _nu)
  {
    return 0.74885397 * std::pow(_eps, -0.0233311) * std::pow(_nu, 0.07192009);
  }
  static Real getTaylorMicroscaleFit(const Real _eps, const Real _nu)
  {
    return 5.35507603 * std::pow(_eps, -1/6.0) * std::sqrt(_nu);
  }
  static Real getHITReynoldsFit(const Real _eps, const Real _nu)
  {
    return 7.33972668 * std::pow(_eps, 1/6.0) / std::sqrt(_nu);
  }
  static Real getTurbKinEnFit(const Real _eps, const Real _nu)
  {
    return 2.81574396 * std::pow(_eps, 2/3.0);
  }

  static Real getKolmogorovL(const Real _eps, const Real _nu)
  {
    return std::pow(_eps, -0.25) * std::pow(_nu, 0.75);
  }
  Real getKolmogorovL() const {
    return getKolmogorovL(dissip_tot > 0? dissip_tot : dissip_visc, nu);
  }

  static Real getKolmogorovT(const Real _eps, const Real _nu)
  {
    return std::sqrt(_nu / _eps);
  }
  Real getKolmogorovT() const {
    return getKolmogorovT(dissip_tot > 0? dissip_tot : dissip_visc, nu);
  }
  Real getKolmogorovU(const Real _eps, const Real _nu) const {
    return std::pow( _eps * _nu , 0.25);
  }

  // Parameters of the histogram
  const int N, nyquist = N/2, nBin = nyquist-1;
  const Real L;

  // Output of the analysis
  Real tke = 0, tke_filtered = 0, tke_prev = 0;
  Real dissip_tot = 0, dissip_visc = 0;
  Real tau_integral = 0, l_integral = 0;
  Real expectedNextTke = 0;
  Real lambda = 0, uprime = 0, Re_lambda = 0;
  Real dt = 0, nu = 0;
  Real * const k_msr;
  Real * const E_msr;
  Real * const Eflux;
};


CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_HITstatistics_h
