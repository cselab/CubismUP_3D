//
//  Cubism3D
//  Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in July 2019.
//

#ifndef CubismUP_3D_ObstacleFromShape_h
#define CubismUP_3D_ObstacleFromShape_h

#include "Obstacle.h"
#include "extra/ObstacleLibrary.h"

#include <utility>
#include <type_traits>

CubismUP_3D_NAMESPACE_BEGIN

/*
 * An obstacle whose shape is defined by a template argument Shape.
 *
 * Shape should implement the following method:
 *
 *    // Returns whether the shape intersects an axis-aligned bounding box.
 *    bool isTouching(std::array<Real, 3> low, std::array<Real, 3> high);
 *
 *    // Signed distance to the shape surface (>0 inside, <0 outside).
 *    Real signedDistance(<array of 3 Reals> position);
 *
 *    // Center of mass velocity.
 *    <array of 3 Reals> comVelocity();
 *
 *    // Local velocity relative to the center-of-mass velocity.
 *    <array of 3 Reals> localRelativeVelocity(<array of 3 Reals> position);
 *
 *    // Factor [0..1] multiplying the lambda,
 *    // used to gradually adding the obstacle to the flow.
 *    Real lambdaFactor();
 *
 *    // Set the current time, for time-dependent shapes.
 *    void setTime(Real);
 */
template <typename Shape>
class ObstacleFromShape : public Obstacle
{
  using position_type = decltype(FluidBlock::min_pos);

  // Expected functions in `Shape`.
  static_assert(std::is_convertible<
      decltype(std::declval<Shape>().isTouching(
          position_type{}, position_type{})), bool>::value);
  static_assert(std::is_convertible<
      decltype(std::declval<Shape>().signedDistance({Real(), Real(), Real()})),
      Real>::value);

  // Here we check only that the velocity result behaves like an array.
  static_assert(std::is_convertible<
      decltype(std::declval<Shape>().comVelocity()[0]), Real>::value);
  static_assert(std::is_convertible<
      decltype(std::declval<Shape>().localRelativeVelocity(
          {Real(), Real(), Real()})[0]), Real>::value);

  static_assert(std::is_convertible<
      decltype(std::declval<Shape>().lambdaFactor(Real())), Real>::value);
  static_assert(std::is_same<
      decltype(std::declval<Shape>().setTime(Real())), void>::value);

public:
  ObstacleFromShape(SimulationData &s,
                    const ObstacleArguments &args,
                    Shape shape) :
      Obstacle(s, args),
      shape_(std::move(shape))
  { }

  void computeVelocities() override
  {
    Obstacle::computeVelocities();

    auto &&v = shape_.comVelocity();
    transVel[0] = transVel_imposed[0] = v[0];
    transVel[1] = transVel_imposed[1] = v[1];
    transVel[2] = transVel_imposed[2] = v[2];
  }

  void create() override
  {
    shape_.setTime(sim.time);
    printf("Cubism step = %d   time = %lg   Uinf = [%lg %lg %lg]\n",
           sim.step, sim.time, sim.uinf[0], sim.uinf[1], sim.uinf[2]);

    // Read the new value of the lambda factor.
    lambda_factor = shape_.lambdaFactor(sim.time);

    const FillBlocks kernel{shape_};
    create_base(kernel);

    const std::vector<cubism::BlockInfo> &vInfo = sim.vInfo();
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < (int)vInfo.size(); ++i) {
      const cubism::BlockInfo &info = vInfo[i];
      if (obstacleBlocks[info.blockID] != nullptr)
        kernel.setVelocity(info, obstacleBlocks[info.blockID]);
    }
  }

  void finalize() override
  {
    // this method allows any computation that requires the char function
    // to be computed. E.g. compute the effective center of mass or removing
    // momenta from udef
  }

private:
  struct FillBlocks : FillBlocksBase<FillBlocks>
  {
    FillBlocks(Shape &shape) : shape_(shape) { }

    bool isTouching(const FluidBlock &b) const
    {
      return shape_.isTouching(b.min_pos, b.max_pos);
    }

    Real signedDistance(const Real x, const Real y, const Real z) const
    {
      return shape_.signedDistance({x, y, z});
    }

    void setVelocity(const cubism::BlockInfo &info,
                     ObstacleBlock * const o) const {
      for (int iz = 0; iz < FluidBlock::sizeZ; ++iz)
      for (int iy = 0; iy < FluidBlock::sizeY; ++iy)
      for (int ix = 0; ix < FluidBlock::sizeX; ++ix) {
        const std::array<Real, 3> p = info.pos<Real>(ix, iy, iz);
        auto &&udef = shape_.localRelativeVelocity(p);

        o->udef[iz][iy][ix][0] = udef[0];
        o->udef[iz][iy][ix][1] = udef[1];
        o->udef[iz][iy][ix][2] = udef[2];
      }
    }

  private:
    Shape &shape_;
  };

  Shape shape_;
};

CubismUP_3D_NAMESPACE_END

#endif
