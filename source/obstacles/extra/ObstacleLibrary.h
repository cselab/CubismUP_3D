//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch) and Wim van Rees.
//

#ifndef CubismUP_3D_ObstacleLibrary_h
#define CubismUP_3D_ObstacleLibrary_h

#include "../../Definitions.h"
#include "Interpolation1D.h"

CubismUP_3D_NAMESPACE_BEGIN

/*
 * A base class for FillBlocks classes.
 *
 * Derived classes should be implemented as (*):
 *      class FillBlocksFOO : FillBlocksBase<FillBlocksFOO> {
 *          (...)
 *      };
 *
 * and are required to implement following member functions:
 *
 * bool isTouching(const BlockInfo &info, int buffer_dx = 0) const;
 *      Returns if the given blocks intersects or touches the object.
 *      False positives are acceptable.
 *
 * Real signedDistance(Real x, Real y, Real z) const;
 *      Returns the signed distance of the given point from the surface of the
 *      object. Positive number stands for inside, negative for outside.
 *
 *
 * (*) That way the base class is able to access functions of the derived class
 *     in compile-time (without using virtual functions). For more info, see:
 *     https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 */

template <typename Derived>
struct FillBlocksBase
{
  using CHIMAT = Real[FluidBlock::sizeZ][FluidBlock::sizeY][FluidBlock::sizeX];
  void operator()(const cubism::BlockInfo &info, ObstacleBlock* const o) const
  {
    // TODO: Remove `isTouching` check and verify that all dependencies are
    //       using this function properly.
    FluidBlock &b = *(FluidBlock *)info.ptrBlock;
    if (!derived()->isTouching(b)) return;
    CHIMAT & __restrict__ SDF = o->sdf;
    for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
    for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
    for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
      Real p[3];
      info.pos(p, ix, iy, iz);
      const Real dist = derived()->signedDistance(p[0], p[1], p[2]);
      SDF[iz][iy][ix] = dist;
      // negative outside of the obstacle, therefore max = minimal distance.
      b(ix,iy,iz).tmpU = std::max(dist, b(ix,iy,iz).tmpU);
    }
  }

private:
  const Derived* derived() const noexcept
  {
    return static_cast<const Derived *>(this);
  }
};

namespace TorusObstacle
{
struct FillBlocks : FillBlocksBase<FillBlocks>
{
  const Real big_r, small_r, h, position[3];
  const Real box[3][2] = {
    {position[0] - 2*(small_r + 2*h),    position[0] + 2*(small_r + 2*h)},
    {position[1] -2*(big_r+small_r+2*h), position[1] +2*(big_r+small_r+2*h)},
    {position[2] -2*(big_r+small_r+2*h), position[2] +2*(big_r+small_r+2*h)}
  };

  FillBlocks(Real _big_r, Real _small_r, Real _h,  Real p[3]) :
    big_r(_big_r), small_r(_small_r), h(_h), position{p[0],p[1],p[2]} { }

  inline bool isTouching(const FluidBlock&b) const
  {
    const Real intersect[3][2] = {
        {std::max(b.min_pos[0], box[0][0]), std::min(b.max_pos[0], box[0][1])},
        {std::max(b.min_pos[1], box[1][0]), std::min(b.max_pos[1], box[1][1])},
        {std::max(b.min_pos[2], box[2][0]), std::min(b.max_pos[2], box[2][1])}
    };
    return intersect[0][1]-intersect[0][0]>0 &&
           intersect[1][1]-intersect[1][0]>0 &&
           intersect[2][1]-intersect[2][0]>0;
  }

  inline Real signedDistance(const Real xo, const Real yo, const Real zo) const
  {
    const Real t[3] = { xo - position[0], yo - position[1], zo - position[2] };
    const Real r = std::sqrt(t[1]*t[1] + t[2]*t[2]);
    if (r > 0) {
      const Real c[3] = { 0, big_r*t[1]/r, big_r*t[2]/r };
      const Real d = std::pow(t[0]-c[0],2) + std::pow(t[1]-c[1],2) + std::pow(t[2]-c[2],2);
      return std::sqrt(d) - small_r;
    }
    else return -1; // very far??? no else in original code: chi = 0
  }
};
}


CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_ObstacleLibrary_h
