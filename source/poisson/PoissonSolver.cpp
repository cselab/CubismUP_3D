//
//  CubismUP_3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PoissonSolver.h"

CubismUP_3D_NAMESPACE_BEGIN
#ifndef CUP_SINGLE_PRECISION
#define MPIREAL MPI_DOUBLE
#else
#define MPIREAL MPI_FLOAT
#endif /* CUP_SINGLE_PRECISION */

void PoissonSolver::_cub2fftw() const
{
  assert(stridez>0 && stridey>0 && stridex>0 && data_size>0);

  //const Real C = sim.bUseStretchedGrid? computeRelativeCorrection_nonUniform()
  //                                    : computeRelativeCorrection();
  //#pragma omp parallel for schedule(static)
  //for(size_t i=0; i<data_size; ++i) data[i] -= std::fabs(data[i])*C;

  #ifndef NDEBUG
  {
    sim.bUseStretchedGrid? computeRelativeCorrection_nonUniform(true)
                         : computeRelativeCorrection(true);
  }
  #endif
}

Real PoissonSolver::computeRelativeCorrection(bool coldrun) const
{
  Real sumRHS = 0, sumABS = 0;
  #pragma omp parallel for schedule(static) reduction(+ : sumRHS, sumABS)
  for(size_t i=0; i<data_size; ++i) {
    sumABS += std::fabs(data[i]); sumRHS += data[i];
  }
  double sums[2] = {sumRHS, sumABS};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, m_comm);
  sums[1] = std::max(std::numeric_limits<double>::epsilon(), sums[1]);
  const Real correction = sums[0] / sums[1];
  if(sim.verbose) {
    //if(coldrun) printf("Integral of RHS after correction:%e\n", sums[0]);
    //else
    printf("Relative RHS correction:%e / %e\n", sums[0], sums[1]);
  }
  return correction;
}

Real PoissonSolver::computeRelativeCorrection_nonUniform(bool coldrun) const
{
  Real sumRHS = 0, sumABS = 0;
  const std::vector<cubism::BlockInfo>& vInfo = sim.vInfo();
  assert(vInfo.size() == local_infos.size());
  #pragma omp parallel for schedule(static) reduction(+ : sumRHS, sumABS)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    const size_t offset = _offset( local_infos[i] );
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++) {
      const size_t src_index = _dest(offset, iz, iy, ix);
      Real h[3]; vInfo[i].spacing(h, ix, iy, iz);
      sumABS += h[0]*h[1]*h[2] * std::fabs( data[src_index] );
      sumRHS += h[0]*h[1]*h[2] * data[src_index];
    }
  }

  double sums[2] = {sumRHS, sumABS};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE,MPI_SUM, m_comm);
  sums[1] = std::max(std::numeric_limits<double>::epsilon(), sums[1]);
  const Real correction = sums[0] / sums[1];
  if(sim.verbose) {
    //if(coldrun) printf("Integral of RHS after correction:%e\n", sums[0]);
    printf("Relative RHS correction:%e / %e\n", sums[0], sums[1]);
  }
  return correction;
}

Real PoissonSolver::computeAverage() const
{
  Real avgP = 0;
  const Real fac = 1.0 / (gsize[0] * gsize[1] * gsize[2]);
  #pragma omp parallel for schedule(static) reduction(+ : avgP)
  for (size_t i = 0; i < data_size; i++) avgP += fac * data[i];
  MPI_Allreduce(MPI_IN_PLACE, &avgP, 1, MPIREAL, MPI_SUM, m_comm);
  return avgP;
}

Real PoissonSolver::computeAverage_nonUniform() const
{
  Real avgP = 0;
  const std::vector<cubism::BlockInfo>& vInfo = sim.vInfo();
  assert(vInfo.size() == local_infos.size());
  #pragma omp parallel for schedule(static) reduction(+ : avgP)
  for(size_t i=0; i<vInfo.size(); ++i)
  {
    const size_t offset = _offset( local_infos[i] );
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++) {
      Real h[3]; vInfo[i].spacing(h, ix, iy, iz);
      const size_t src_index = _dest(offset, iz, iy, ix);
      avgP += h[0]*h[1]*h[2] * data[src_index];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &avgP, 1, MPIREAL, MPI_SUM, m_comm);
  avgP /= sim.extent[0] * sim.extent[1] * sim.extent[2];
  return avgP;
}

void PoissonSolver::_fftw2cub() const
{
  const size_t NlocBlocks = local_infos.size();
  #pragma omp parallel for schedule(static)
  for(size_t i=0; i<NlocBlocks; ++i) {
    BlockType& b = *(BlockType*) local_infos[i].ptrBlock;
    const size_t offset = _offset( local_infos[i] );
    for(int iz=0; iz<BlockType::sizeZ; iz++)
    for(int iy=0; iy<BlockType::sizeY; iy++)
    for(int ix=0; ix<BlockType::sizeX; ix++) {
      const size_t src_index = _dest(offset, iz, iy, ix);
      b(ix,iy,iz).p = data[src_index];
    }
  }
}

void PoissonSolver::reset() const
{
  memset(data, 0, data_size * sizeof(Real));
}

CubismUP_3D_NAMESPACE_END
