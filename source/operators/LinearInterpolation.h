//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

#ifndef CUBISMUP_3D_LINEAR_INTERPOLATION_H
#define CUBISMUP_3D_LINEAR_INTERPOLATION_H

#include "Operator.h"
#include <vector>

CubismUP_3D_NAMESPACE_BEGIN

namespace detail {

class LinearCellCenteredInterpolation;

/* Interpolate values for points contained in a single block. */
template <typename Getter, typename Setter>
class Kernel
{
public:
  static constexpr std::array<int, 3> stencil_start{-1, -1, -1};
  static constexpr std::array<int, 3> stencil_end{2, 2, 2};
  const cubism::StencilInfo stencil;

  Kernel(const LinearCellCenteredInterpolation &_owner,
         Getter &_getter,
         Setter &_setter,
         std::vector<int> components)
      : stencil(-1, -1, -1, 2, 2, 2, true, std::move(components)),
        owner(_owner),
        getter(_getter),
        setter(_setter)
  { }

  template <typename Lab, typename BlockType>
  void operator()(Lab &lab, const cubism::BlockInfo &info, BlockType &o) const;

private:
  const LinearCellCenteredInterpolation &owner;
  Getter &getter;
  Setter &setter;
};


/* Distribute points over blocks. */
class LinearCellCenteredInterpolation : public Operator
{
public:
  LinearCellCenteredInterpolation(SimulationData &s) : Operator(s) { }

  int get_cell_index(double x, int d) const noexcept
  {
    // We assume non-periodic boundaries and that the query points will not be
    // very close to the boundary.
    const int id = (int)(CN_over_extent[d] * x - 0.5);
    return std::max(0, std::min(id, CN[d] - 2));
  }

  std::array<int, 3> get_cell_index(const double p[3]) const noexcept
  {
    return {
      get_cell_index(p[0], 0),
      get_cell_index(p[1], 1),
      get_cell_index(p[2], 2),
    };
  }

  template <typename Array, typename Getter, typename Setter>
  void interpolate(const Array &points,
                   Getter &getter,
                   Setter &setter,
                   std::vector<int> components)
  {
    typedef typename FluidGridMPI::BlockType Block;
    N[0] = grid->getBlocksPerDimension(0);
    N[1] = grid->getBlocksPerDimension(1);
    N[2] = grid->getBlocksPerDimension(2);
    CN[0] = N[0] * Block::sizeX;
    CN[1] = N[1] * Block::sizeY;
    CN[2] = N[2] * Block::sizeZ;
    CN_over_extent[0] = CN[0] / sim.extent[0];
    CN_over_extent[1] = CN[1] / sim.extent[1];
    CN_over_extent[2] = CN[2] / sim.extent[2];
    particles.resize(N[0] * N[1] * N[2]);

    // Map particles to the blocks.
    for (int i = 0; i < (int)points.size(); ++i) {
      const auto &point = points[i];
      Particle part;
      part.id = i;
      part.pos[0] = point[0];
      part.pos[1] = point[1];
      part.pos[2] = point[2];

      const std::array<int, 3> cell_index = get_cell_index(part.pos);
      const int block_index[3] = {
        cell_index[0] / Block::sizeX,
        cell_index[1] / Block::sizeY,
        cell_index[2] / Block::sizeZ,
      };
      const int idx = block_index[0] + N[0] * (block_index[1] + N[1] * block_index[2]);

      particles[idx].push_back(part);
    }

    Kernel<decltype(getter), decltype(setter)>
        kernel(*this, getter, setter, std::move(components));
    compute(kernel);
  }

  // GenericCoordinator stuff we don't care about now.
  void operator()(const double /* dt */) override { abort(); }
  std::string getName(void) { return "LinearCellCenteredInterpolation"; }

  struct Particle {
    int id;
    double pos[3];
  };

  int N[3];
  int CN[3];
  double CN_over_extent[3];
  std::vector<std::vector<Particle>> particles;
};


template <typename Getter, typename Setter>
template <typename Lab, typename BlockType>
void Kernel<Getter, Setter>::operator()(
    Lab &lab,
    const cubism::BlockInfo &info,
    BlockType &o) const
{
  typedef typename FluidGridMPI::BlockType Block;
  const int block_index = info.index[0] + owner.N[0] * (
                          info.index[1] + owner.N[1] * info.index[2]);
  const double invh = 1.0 / info.h_gridpoint;

  for (const auto &part : owner.particles[block_index]) {
    // Global cell index.
    const std::array<int, 3> cell_index = owner.get_cell_index(part.pos);

    // Block-level cell index.
    const int idx[3] = {
      cell_index[0] - info.index[0] * Block::sizeX,
      cell_index[1] - info.index[1] * Block::sizeY,
      cell_index[2] - info.index[2] * Block::sizeZ,
    };

    const double ipos[3] = {
      invh * (part.pos[0] - info.origin[0]),
      invh * (part.pos[1] - info.origin[1]),
      invh * (part.pos[2] - info.origin[2]),
    };

    // Compute 1D weights.
    const double w[3] = {
      ipos[0] - idx[0] - 0.5,
      ipos[1] - idx[1] - 0.5,
      ipos[2] - idx[2] - 0.5,
    };

    // Do M2P interpolation.
    const double w000 = (1 - w[0]) * (1 - w[1]) * (1 - w[2]);
    const double w010 = (1 - w[0]) * (    w[1]) * (1 - w[2]);
    const double w100 = (    w[0]) * (1 - w[1]) * (1 - w[2]);
    const double w110 = (    w[0]) * (    w[1]) * (1 - w[2]);
    const double w001 = (1 - w[0]) * (1 - w[1]) * (    w[2]);
    const double w011 = (1 - w[0]) * (    w[1]) * (    w[2]);
    const double w101 = (    w[0]) * (1 - w[1]) * (    w[2]);
    const double w111 = (    w[0]) * (    w[1]) * (    w[2]);
    setter(part.id,
           w000 * getter(lab.read(idx[0]    , idx[1]    , idx[2]    ))
         + w010 * getter(lab.read(idx[0]    , idx[1] + 1, idx[2]    ))
         + w100 * getter(lab.read(idx[0] + 1, idx[1]    , idx[2]    ))
         + w110 * getter(lab.read(idx[0] + 1, idx[1] + 1, idx[2]    ))
         + w001 * getter(lab.read(idx[0]    , idx[1]    , idx[2] + 1))
         + w011 * getter(lab.read(idx[0]    , idx[1] + 1, idx[2] + 1))
         + w101 * getter(lab.read(idx[0] + 1, idx[1]    , idx[2] + 1))
         + w111 * getter(lab.read(idx[0] + 1, idx[1] + 1, idx[2] + 1)));
  }
}

}  // namespace _linint_impl


/*
 * Cell-centered mesh to particle linear interpolation.
 *
 * For each given point, interpolate the value of the field.
 *
 * Arguments:
 *   - points - Array of the points, where points support operator [] for
 *              accessing x, y, z coordinates.
 *   - getter - Lambda of a single argument (BlockLab), returning the value
 *              to be interpolated. The value should support addition and
 *              multiplication by a scalar. If you are interpolating more than
 *              one value simultaneously, check `utils/ScalarArray.h`.
 *   - setter - Lambda function of two arguments (point ID, interpolated value).
 *   - components - Stencil components.
 */
template <typename Array, typename Getter, typename Setter>
void linearCellCenteredInterpolation(
    SimulationData &sim,
    const Array &points,
    Getter&& getter,
    Setter&& setter,
    std::vector<int> components)
{
  detail::LinearCellCenteredInterpolation I{sim};
  I.interpolate(points, getter, setter, std::move(components));
}


CubismUP_3D_NAMESPACE_END

#endif
