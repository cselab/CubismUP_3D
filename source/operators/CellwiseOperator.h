//
//  Cubism3D
//  Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch).
//

/*
 * This file contains helper functions for applying single-cell or
 * stencil-based kernels on the grid.
 */
#ifndef CUBISMUP3D_CELLWISE_OPERATOR_H
#define CUBISMUP3D_CELLWISE_OPERATOR_H

#include "Operator.h"

CubismUP_3D_NAMESPACE_BEGIN

/*
 * Struct passed to kernels in `apply_kernel` and `apply_stencil_kernel` functions.
 *
 * Ideally, we would put all interesting quantities as struct members and let
 * compiler optimize away unused ones. Although some basic tests show that
 * compiler indeed do so, it is not sure if that holds for arbitrarily large
 * structs.
 *
 * Thus, we put only the most necessary items in the struct and provide other
 * values as member functions.
 */
struct CellInfo
{
  const cubism::BlockInfo &block_info;
  int ix, iy, iz;

  std::array<Real, 3> get_pos() const { return block_info.pos<Real>(ix, iy, iz); }
  int get_abs_ix() const { return ix + block_info.index[0] * FluidBlock::sizeX; }
  int get_abs_iy() const { return iy + block_info.index[1] * FluidBlock::sizeY; }
  int get_abs_iz() const { return iz + block_info.index[2] * FluidBlock::sizeZ; }
};


/*
 * Apply a given single-cell kernel to each cell of the grid.
 *
 * Usage example:
 *    applyKernel(sim, [](CellInfo info, FluidElement &e) {
 *      e.u = e.tmpU;
 *      e.v = e.tmpV;
 *      e.w = e.tmpW;
 *    });
 */
template <typename Func>
void applyKernel(SimulationData &sim, Func func)
{
  const std::vector<cubism::BlockInfo> &vInfo = sim.vInfo();
  int size = (int)vInfo.size();

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < size; ++i) {
    FluidBlock &b = *(FluidBlock *)vInfo[i].ptrBlock;
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      CellInfo info{vInfo[i], ix, iy, iz};
      func(info, b(ix, iy, iz));
    }
  }
}


/*
 * Lab wrapper that shifts from cell-based indices (given by the user) to the
 * block-based indices (required by the original BlockLab).
 */
struct StencilKernelLab {
  LabMPI &lab;
  int ix, iy, iz;

  FluidElement& operator()(int dx, int dy = 0, int dz = 0) const {
    return lab(ix + dx, iy + dy, iz + dz);
  }
};

/*
 * Apply a given stencil kernel to each cell of the grid.
 *
 * Usage example:
 *    Real factor = 0.5 / h;
 *    applyStencilKernel(
 *      sim,
 *      StencilInfo{-1, 0, 0, 2, 1, 1, false, 1, FE_U},
 *      [factor](StencilKernelLab lab, CellInfo info, FluidElement &out) {
 *        out.df = factor * (lab(1, 0, 0).f - lab(-1, 0, 0).f);
 *    });
 */
template <typename Func>
void applyStencilKernel(SimulationData &sim, cubism::StencilInfo stencil, Func func)
{
  // Block-based kernel.
  struct Kernel {
    const cubism::StencilInfo stencil;
    Func func;

    void operator()(LabMPI &lab, const cubism::BlockInfo &block_info, FluidBlock &out) const
    {
      for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
      for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
      for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
        StencilKernelLab lab_wrapper{lab, ix, iy, iz};
        CellInfo info{block_info, ix, iy, iz};
        func(lab_wrapper, info, out(ix, iy, iz));
      }
    }
  };

  struct CellwiseOperator : Operator {
    Kernel kernel;

    CellwiseOperator(SimulationData &s, const cubism::StencilInfo &stencil, Func func)
        : Operator(s), kernel{stencil, func} {}

    void operator()(const Real /* dt */) {
      // For now we ignore the `dt` argument. We could e.g. pass it via the
      // `CellInfo` struct. In that case, we would need to rename it to
      // something like `Extra`.
      compute(kernel);
    }

    std::string getName() { return "apply_stencil_kernel::CellwiseOperator"; }
  };

  CellwiseOperator op{sim, stencil, func};
  op(0.0);  // dt is unused for now.
}

CubismUP_3D_NAMESPACE_END

#endif
