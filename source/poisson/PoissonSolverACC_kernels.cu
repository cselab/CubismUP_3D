//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

//#include "PoissonSolverScalarFFTW_ACC.h"
//#include <cuda_runtime_api.h>

#include "../Base.h"
#include "PoissonSolverACC_common.h"

#include <cassert>
#include <cmath>
#include <cufft.h>
#include <limits>

using Real = cubismup3d::Real;

__global__
void _poisson_spectral_kernel(acc_c*const __restrict__ out,
  const long Gx, const long Gy, const long Gz,
  const long nx, const long ny, const long nz, const long nz_hat,
  const long sx, const long sy, const long sz,
  const Real wx, const Real wy, const Real wz, const Real fac)
{
  const long k = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long i = blockDim.z * blockIdx.z + threadIdx.z;
  if( i >= nx || j >= ny || k >= nz_hat ) return;

  const long kx = sx + i, ky = sy + j, kz = sz + k;
  const long kkx = kx > Gx/2 ? kx-Gx : kx;
  const long kky = ky > Gy/2 ? ky-Gy : ky;
  const long kkz = kz > Gz/2 ? kz-Gz : kz;
  const Real rkx = kkx*wx, rky = kky*wy, rkz = kkz*wz;
  const Real kinv = kkx||kky||kkz? - fac/(rkx*rkx + rky*rky + rkz*rkz) : 0;
  const long index = i*nz_hat*ny + j*nz_hat + k;
  out[index][0] *= kinv;
  out[index][1] *= kinv;
}

__global__
void _poisson_findiff_kernel(acc_c*const __restrict__ out,
  const long Gx, const long Gy, const long Gz,
  const long nx, const long ny, const long nz, const long nz_hat,
  const long sx, const long sy, const long sz,
  const Real wx, const Real wy, const Real wz, const Real fac)
{
  const long k = blockDim.x * blockIdx.x + threadIdx.x;
  const long j = blockDim.y * blockIdx.y + threadIdx.y;
  const long i = blockDim.z * blockIdx.z + threadIdx.z;
  if( i >= nx || j >= ny || k >= nz_hat ) return;

  const long kx = sx + i, ky = sy + j, kz = sz + k;
  const Real denom = 2*(std::cos(wx*kx) +std::cos(wy*ky) +std::cos(wz*kz)) - 6;
  static constexpr Real EPS = 2.2204460492503131E-16;
  const Real solutionFactor = denom < -EPS? fac / denom : 0;
  const long index = i*nz_hat*ny + j*nz_hat + k;
  out[index][0] *= solutionFactor;
  out[index][1] *= solutionFactor;
}

void _fourier_filter_gpu(acc_c*const __restrict__ data_hat,
 const size_t gsize[3],const int osize[3] , const int ostart[3], const Real h)
{
  int blocksInX = std::ceil(osize[2] / 16.);
  int blocksInY = std::ceil(osize[1] / 8.);
  int blocksInZ = std::ceil(osize[0] / 1.);
  const int nz_hat = gsize[2]/2 + 1;
  dim3 Dg(blocksInX, blocksInY, blocksInZ), Db(16, 8, 1);
  // RHS comes into this function premultiplied by h^3 (as in FMM):
  if(1)
  {
    // Solution has to be normalized (1/N^3) and multiplied by Laplace op finite
    // diffs coeff. We use finite diffs consistent with press proj, therefore
    // +/- 2h, and Poisson coef is (2h)^2. Due to RHS premultiplied by h^3:
    const Real norm = 4.0 / ( gsize[0]*h * gsize[1] * gsize[2] );
    const Real wfac[3] = {
      Real(4*M_PI)/(gsize[0]), Real(4*M_PI)/(gsize[1]), Real(4*M_PI)/(gsize[2])
    };
    _poisson_findiff_kernel<<<Dg, Db>>>(data_hat,
                      gsize[0],  gsize[1],  gsize[2],
                      osize[0],  osize[1],  osize[2], nz_hat,
                     ostart[0], ostart[1], ostart[2],
                       wfac[0],   wfac[1],   wfac[2], norm);
    // const long lastI = (long) gsize[0]/2 - ostart[0];
    // const long lastJ = (long) gsize[1]/2 - ostart[1];
    // const long lastK = (long) gsize[2]/2 - ostart[2];
    // if(ostart[0] == 0 &&
    //    ostart[1] == 0) {
    //   const size_t idWSF = (0 * osize[1] + 0)*nz_hat + 0;
    //   const size_t idWSB = (0 * osize[1] + 0)*nz_hat + lastK;
    //   cudaMemset(data_hat + idWSF, 0, 2 * sizeof(Real));
    //   cudaMemset(data_hat + idWSB, 0, 2 * sizeof(Real));
    // }
    // if(ostart[0] == 0 &&
    //    ostart[1] <= gsize[1]/2 && ostart[1]+osize[1] > gsize[1]/2) {
    //   const size_t idWNF = (0 * osize[1] + lastJ)*nz_hat + 0;
    //   const size_t idWNB = (0 * osize[1] + lastJ)*nz_hat + lastK;
    //   cudaMemset(data_hat + idWNF, 0, 2 * sizeof(Real));
    //   cudaMemset(data_hat + idWNB, 0, 2 * sizeof(Real));
    // }
    // if(ostart[0] <= gsize[0]/2 && ostart[0]+osize[0] > gsize[0]/2 &&
    //    ostart[1] == 0) {
    //   const size_t idESF = (lastI * osize[1] + 0)*nz_hat + 0;
    //   const size_t idESB = (lastI * osize[1] + 0)*nz_hat + lastK;
    //   cudaMemset(data_hat + idESF, 0, 2 * sizeof(Real));
    //   cudaMemset(data_hat + idESB, 0, 2 * sizeof(Real));
    // }
    // if(ostart[0] <= gsize[0]/2 && ostart[0]+osize[0] > gsize[0]/2 &&
    //    ostart[1] <= gsize[1]/2 && ostart[1]+osize[1] > gsize[1]/2) {
    //   const size_t idENF = (lastI * osize[1] + lastJ)*nz_hat + 0;
    //   const size_t idENB = (lastI * osize[1] + lastJ)*nz_hat + lastK;
    //   cudaMemset(data_hat + idENF, 0, 2 * sizeof(Real));
    //   cudaMemset(data_hat + idENB, 0, 2 * sizeof(Real));
    // }
  }
  else
  {
    const Real norm = 1.0 / ( gsize[0]*h * gsize[1]*h * gsize[2]*h );
    const Real wfac[3] = {
      2*M_PI/(h*gsize[0]), 2*M_PI/(h*gsize[1]), 2*M_PI/(h*gsize[2])
    };
    _poisson_spectral_kernel<<<Dg, Db>>>(data_hat,
                      gsize[0],  gsize[1],  gsize[2],
                      osize[0],  osize[1],  osize[2], nz_hat,
                     ostart[0], ostart[1], ostart[2],
                       wfac[0],   wfac[1],   wfac[2], norm);
    //if(ostart[0]==0 && ostart[1]==0 && ostart[2]==0)
    //  cudaMemset(data_hat, 0, 2 * sizeof(Real));
  }
}

__global__ void kPos(const int iSzX, const int iSzY, const int iSzZ,
  const int iStX, const int iStY, const int iStZ, const int nGlobX,
  const int nGlobY, const int nGlobZ, const size_t nZpad, Real*const in_out)
{
  const int k = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int i = blockDim.z * blockIdx.z + threadIdx.z;
  if ( (i >= iSzX) || (j >= iSzY) || (k >= 2*nZpad) ) return;
  const size_t linidx = k + 2*nZpad*(j + iSzY*i);
  const Real I = i + iStX, J = j + iStY, K = k + iStZ;
  in_out[linidx] = K + nGlobZ * (J + nGlobY * I);
}

__global__ void kGreen(const int iSzX, const int iSzY, const int iSzZ,
  const int iStX, const int iStY, const int iStZ,
  const int nGlobX, const int nGlobY, const int nGlobZ, const size_t nZpad,
  const Real h, Real*const in_out)
{
  const int k = blockDim.x * blockIdx.x + threadIdx.x;
  const int j = blockDim.y * blockIdx.y + threadIdx.y;
  const int i = blockDim.z * blockIdx.z + threadIdx.z;
  if ( (i >= iSzX) || (j >= iSzY) || (k >= 2*nZpad) ) return;
  const size_t linidx = k + 2*nZpad*(j + iSzY*i);
  const int I = i + iStX, J = j + iStY, K = k + iStZ;
  const Real xi = I>=nGlobX? 2*nGlobX-1 - I : I;
  const Real yi = J>=nGlobY? 2*nGlobY-1 - J : J;
  const Real zi = K>=nGlobZ? 2*nGlobZ-1 - K : K;
  const Real r = std::sqrt(xi*xi + yi*yi + zi*zi);
  if(r > 0) in_out[linidx] = - h * h / ( 4 * M_PI * r );
  // G = r_eq^2 / 2 = std::pow(3/8/pi/sqrt(2))^(2/3) * h^2 :
  else      in_out[linidx] = - Real(0.1924173658) * h * h;
}

__global__ void kCopyC2R(const int oSzX,const int oSzY,const int oSzZ,
  const Real norm, const size_t nZpad, const acc_c*const G_hat, Real*const m_kernel)
{
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  const int i = threadIdx.z + blockIdx.z * blockDim.z;
  if ( (i >= oSzX) || (j >= oSzY) || (k >= nZpad) ) return;
  const size_t linidx = (i*oSzY + j)*nZpad + k;
  m_kernel[linidx] = G_hat[linidx][0] * norm;
}

__global__ void kFreespace(const int oSzX, const int oSzY, const int oSzZ,
  const size_t nZpad, const Real*const G_hat, acc_c*const in_out)
{
  const int k = threadIdx.x + blockIdx.x * blockDim.x;
  const int j = threadIdx.y + blockIdx.y * blockDim.y;
  const int i = threadIdx.z + blockIdx.z * blockDim.z;
  if ( (i >= oSzX) || (j >= oSzY) || (k >= nZpad) ) return;
  const size_t linidx = (i*oSzY + j)*nZpad + k;
  in_out[linidx][0] *= G_hat[linidx];
  in_out[linidx][1] *= G_hat[linidx];
}

void dSolveFreespace(const int ox,const int oy,const int oz,const size_t mz_pad,
  const Real*const G_hat, Real*const gpu_rhs)
{
  dim3 dB(16, 8, 1);
  dim3 dG(std::ceil(oz/16.0), std::ceil(oy/8.0), ox);
  kFreespace <<<dG, dB>>> (ox,oy,oz, mz_pad, G_hat, (acc_c*) gpu_rhs);
}

void initGreen(const int *isz, const int *ist,
  int nx, int ny, int nz, const Real h, Real*const gpu_rhs)
{
  const int mz = 2*nz -1, mz_pad = mz/2 +1;
  dim3 dB(16, 8, 1);
  dim3 dG(std::ceil(isz[2]/16.0), std::ceil(isz[1]/8.0), isz[0]);

  kGreen<<<dG, dB>>> (isz[0],isz[1],isz[2], ist[0],ist[1],ist[2],
    nx, ny, nz, mz_pad, h, gpu_rhs);
}

void realGreen(const int*osz, const int*ost, int nx, int ny, int nz,
  const Real h, Real*const m_kernel, Real*const gpu_rhs)
{
  const int mx = 2*nx -1, my = 2*ny -1, mz = 2*nz -1, mz_pad = mz/2 +1;
  const Real norm = 1.0 / (mx*h * my*h * mz*h);
  dim3 dB(16, 8, 1);
  dim3 dG(std::ceil(osz[2]/16.0), std::ceil(osz[1]/8.0), osz[0]);

  kCopyC2R<<<dG, dB>>> (osz[0],osz[1],osz[2], norm, mz_pad,
    (acc_c*)gpu_rhs, m_kernel);

  //{
  //  dim3 dB(16, 8, 1);
  //  dim3 dG(std::ceil(isz[2]/16.0), std::ceil(isz[1]/8.0), isz[0]);
  //  kPos<<<dG, dB>>> (isz[0],isz[1],isz[2], ist[0],ist[1],ist[2], mx,my,mz, mz_pad, gpu_rhs);
  //}
}

#if __CUDACC_VER_MAJOR__ >= 9
#define MASK_ALL_WARP 0xFFFFFFFF
#define warpShflDown(var, delta)   __shfl_down_sync (MASK_ALL_WARP, var, delta)
#else
#define warpShflDown(var, delta)   __shfl_down (var, delta)
#endif


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    using ULLI_t = unsigned long long int;
    ULLI_t* address_as_ull = (ULLI_t*)address;
    ULLI_t old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__device__ inline uint32_t __laneid()
{
    uint32_t laneid;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

static constexpr long NLOOPKERNEL = 16;

__global__
void _analysis_filter_kernel( acc_c*const __restrict__ Uhat,
  acc_c*const __restrict__ Vhat, acc_c*const __restrict__ What,
  const long Gx, const long Gy, const long Gz,
  const long nx, const long ny, const long nz, const long nz_hat,
  const long sx, const long sy, const long sz,
  const Real wx, const Real wy, const Real wz, const Real h,
  const Real nyquist, const Real nyquist_scaling, Real * const reductionBuf)
{
  const long k = blockDim.x * blockIdx.x + threadIdx.x;
  const long J = blockDim.y * blockIdx.y + threadIdx.y;
  const long i = blockDim.z * blockIdx.z + threadIdx.z;
  Real tke = 0, eps = 0, lInt = 0, tkeFiltered = 0;

  if( i < nx && k < nz_hat )
  for(long j = J * NLOOPKERNEL; j < (J+1) * NLOOPKERNEL && j < ny; ++j)
  {
    const long ind = i*nz_hat*ny + j*nz_hat + k;
    const long kx = sx + i, ky = sy + j, kz = sz + k;
    const long kkx = kx > Gx/2 ? kx-Gx : kx;
    const long kky = ky > Gy/2 ? ky-Gy : ky;
    const long kkz = kz > Gz/2 ? kz-Gz : kz;
    const Real rkx = kkx * wx, rky = kky * wy, rkz = kkz * wz;
    const Real k2 = rkx * rkx + rky * rky + rkz * rkz;
    const Real mult = (kz==0) or (kz==Gz/2) ? 1 : 2;

    const Real dXfac = 2.0 * std::sin(h * rkz);
    const Real dYfac = 2.0 * std::sin(h * rky);
    const Real dZfac = 2.0 * std::sin(h * rkx);
    const Real UR = Uhat[ind][0], UI = Uhat[ind][1];
    const Real VR = Vhat[ind][0], VI = Vhat[ind][1];
    const Real WR = What[ind][0], WI = What[ind][1];
    const Real dUdYR = - UI * dYfac, dUdYI = UR * dYfac;
    const Real dUdZR = - UI * dZfac, dUdZI = UR * dZfac;
    const Real dVdXR = - VI * dXfac, dVdXI = VR * dXfac;
    const Real dVdZR = - VI * dZfac, dVdZI = VR * dZfac;
    const Real dWdXR = - WI * dXfac, dWdXI = WR * dXfac;
    const Real dWdYR = - WI * dYfac, dWdYI = WR * dYfac;
    const Real OMGXR = dWdYR - dVdZR, OMGXI = dWdYI - dVdZI;
    const Real OMGYR = dUdZR - dWdXR, OMGYI = dUdZI - dWdXI;
    const Real OMGZR = dVdXR - dUdYR, OMGZI = dVdXI - dUdYI;

    const Real E = mult/2 * (UR*UR + UI*UI + VR*VR + VI*VI + WR*WR + WI*WI);

    tke  += E; // Total kinetic energy
    //eps  += mult/2 * E *  k2;
    eps  += mult/2 * ( OMGXR*OMGXR + OMGXI*OMGXI
                     + OMGYR*OMGYR + OMGYI*OMGYI
                     + OMGZR*OMGZR + OMGZI*OMGZI);    // Dissipation rate
    lInt += (k2 > 0) ? E / std::sqrt(k2) : 0; // Large eddy length scale

    const long kind = kkx * kkx + kky * kky + kkz * kkz;
    if (kind < nyquist*nyquist) {
      const int binID = std::floor(std::sqrt((Real) kind) * nyquist_scaling);
      //assert(binID < nBins);
      // reduction buffer here holds also the energy spectrum, shifted by 3
      // to hold also tke, eps and tau
      atomicAdd(reductionBuf + 4 + binID, E);
    }
    if (k2 > 0 && kind < nyquist*nyquist) {
      tkeFiltered += E;
    } else {
      Uhat[ind][0] = 0; Uhat[ind][1] = 0;
      Vhat[ind][0] = 0; Vhat[ind][1] = 0;
      What[ind][0] = 0; What[ind][1] = 0;
    }
  }

  #pragma unroll
  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    tke         = tke         + warpShflDown(        tke, offset);
    eps         = eps         + warpShflDown(        eps, offset);
    lInt        = lInt        + warpShflDown(       lInt, offset);
    tkeFiltered = tkeFiltered + warpShflDown(tkeFiltered, offset);
  }

  if (__laneid() == 0) { // thread 0 does the only atomic sum
    atomicAdd(reductionBuf + 0, tke);
    atomicAdd(reductionBuf + 1, eps);
    atomicAdd(reductionBuf + 2, lInt);
    atomicAdd(reductionBuf + 3, tkeFiltered);
  }
}

void _compute_HIT_analysis(
  acc_c*const Uhat, acc_c*const Vhat, acc_c*const What,
  const size_t gsize[3], const int osize[3] , const int ostart[3], const Real h,
  Real& tke, Real& eps, Real& lInt, Real& tkeFiltered, Real * const eSpectrum,
  const size_t nBins, const Real nyquist
)
{
  Real * rdxBuf;
  // single buffer to contain both eps, tke, lInt and spectrum
  cudaMalloc((void**)& rdxBuf, (4+nBins) * sizeof(Real));
  cudaMemset(rdxBuf, 0, (4+nBins) * sizeof(Real));

  const Real wfac[3] = {
    2*M_PI/(h*gsize[0]), 2*M_PI/(h*gsize[1]), 2*M_PI/(h*gsize[2])
  };
  //printf("wfac[3] %e %e %e\n", wfac[0], wfac[1], wfac[2]);
  const int nz_hat = gsize[2]/2 + 1;
  const Real nyquist_scaling = (nyquist-1) / (Real) nyquist;
  int blocksInX = std::ceil(nz_hat   / 16.0 );
  int blocksInY = std::ceil(osize[1] / (NLOOPKERNEL * 1.0) );
  int blocksInZ = std::ceil(osize[0] / 4.0 );
  dim3 Dg(blocksInX, blocksInY, blocksInZ), Db(16, 1, 4);
  assert(gsize[2] == osize[2]);
  _analysis_filter_kernel<<<Dg, Db>>>( Uhat, Vhat, What,
     gsize[0], gsize[1], gsize[2], osize[0],osize[1],osize[2], nz_hat,
    ostart[0],ostart[1],ostart[2],  wfac[0], wfac[1], wfac[2],
    h, nyquist, nyquist_scaling, rdxBuf);

  cudaMemcpy(&tke,        rdxBuf+0,       sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(&eps,        rdxBuf+1,       sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(&lInt,       rdxBuf+2,       sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(&tkeFiltered,rdxBuf+3,       sizeof(Real), cudaMemcpyDeviceToHost);
  cudaMemcpy(eSpectrum,   rdxBuf+4, nBins*sizeof(Real), cudaMemcpyDeviceToHost);
  cudaFree(rdxBuf);
}
