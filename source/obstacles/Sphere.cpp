//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Sphere.h"
#include "extra/ObstacleLibrary.h"

#include <Cubism/ArgumentParser.h>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace SphereObstacle
{
struct FillBlocks : FillBlocksBase<FillBlocks>
{
  const Real radius, h, safety = (2+SURFDH)*h;
  const double position[3];
  const Real box[3][2] = {
    {(Real)position[0] - (radius+safety), (Real)position[0] + (radius+safety)},
    {(Real)position[1] - (radius+safety), (Real)position[1] + (radius+safety)},
    {(Real)position[2] - (radius+safety), (Real)position[2] + (radius+safety)}
  };

  FillBlocks(const Real _radius, const Real max_dx, const double pos[3]):
  radius(_radius), h(max_dx), position{pos[0],pos[1],pos[2]} {}

  inline bool isTouching(const FluidBlock&b) const
  {
    const Real intersect[3][2] = {
      {std::max(b.min_pos[0], box[0][0]), std::min(b.max_pos[0], box[0][1]) },
      {std::max(b.min_pos[1], box[1][0]), std::min(b.max_pos[1], box[1][1]) },
      {std::max(b.min_pos[2], box[2][0]), std::min(b.max_pos[2], box[2][1]) }
    };
    return intersect[0][1]-intersect[0][0]>0 &&
           intersect[1][1]-intersect[1][0]>0 &&
           intersect[2][1]-intersect[2][0]>0;
  }

  inline Real signedDistance(const Real x, const Real y, const Real z) const
  {
    const Real dx = x-position[0], dy = y-position[1], dz = z-position[2];
    return radius - std::sqrt(dx*dx + dy*dy + dz*dz); // pos inside, neg outside
  }
};
}

namespace HemiSphereObstacle
{
struct FillBlocks : FillBlocksBase<FillBlocks>
{
  const Real radius, h, safety = (2+SURFDH)*h;
  const double position[3];
  const Real box[3][2] = {
    {(Real)position[0] -radius -safety, (Real)position[0] +safety},
    {(Real)position[1] -radius -safety, (Real)position[1] +radius +safety},
    {(Real)position[2] -radius -safety, (Real)position[2] +radius +safety}
  };

  FillBlocks(const Real _radius, const Real max_dx, const double pos[3]):
  radius(_radius), h(max_dx), position{pos[0],pos[1],pos[2]} {}

  inline bool isTouching(const FluidBlock&b) const
  {
    const Real intersect[3][2] = {
      { std::max(b.min_pos[0], box[0][0]), std::min(b.max_pos[0], box[0][1]) },
      { std::max(b.min_pos[1], box[1][0]), std::min(b.max_pos[1], box[1][1]) },
      { std::max(b.min_pos[2], box[2][0]), std::min(b.max_pos[2], box[2][1]) }
    };
    return intersect[0][1]-intersect[0][0]>0 &&
           intersect[1][1]-intersect[1][0]>0 &&
           intersect[2][1]-intersect[2][0]>0;
  }

  inline Real signedDistance(const Real x, const Real y, const Real z) const
  { // pos inside, neg outside
    const Real dx = x-position[0], dy = y-position[1], dz = z-position[2];
    return std::min( -dx, radius -std::sqrt(dx*dx + dy*dy + dz*dz) );
  }
};
}

Sphere::Sphere(SimulationData& s, ArgumentParser& p)
    : Obstacle(s, p), SphereArguments(0.5 * length)
{
  accel_decel = p("-accel").asBool(false);
  bHemi = p("-hemisphere").asBool(false);
  if(accel_decel) {
    if(not bForcedInSimFrame[0]) {
      printf("Warning: sphere was not set to be forced in x-dir, yet the accel_decel pattern is active.\n");
    }
    umax = p("-xvel").asDouble(0.0);
    tmax = p("-T").asDouble(1.);
  }
}


Sphere::Sphere(
    SimulationData& s,
    ObstacleArguments &args,
    const double R)
    : Obstacle(s, args), SphereArguments(R) { }


Sphere::Sphere(
    SimulationData& s,
    ObstacleArguments &args,
    const double R,
    const double _umax,
    const double _tmax)
    : Obstacle(s, args), SphereArguments(R)
{
  umax = _umax;
  tmax = _tmax;
}

Sphere::Sphere(
    SimulationData &s,
    const ObstacleAndSphereArguments &args)
    : Obstacle(s, args), SphereArguments(args) { }  // Object slicing.


void Sphere::create()
{
  const Real h = sim.maxH();
  if(bHemi) {
    const HemiSphereObstacle::FillBlocks K(radius, h, position);
    create_base<HemiSphereObstacle::FillBlocks>(K);
  } else {
    const SphereObstacle::FillBlocks K(radius, h, position);
    create_base<SphereObstacle::FillBlocks>(K);
  }
}

void Sphere::finalize()
{
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}


void Sphere::computeVelocities()
{
  if(accel_decel) {
    if(sim.time<tmax)
      transVel_imposed[0] = umax*sim.time/tmax;
    else if (sim.time<2*tmax)
      transVel_imposed[0] = umax*(2*tmax-sim.time)/tmax;
    else
      transVel_imposed[0] = 0;
  }

  Obstacle::computeVelocities();
}

CubismUP_3D_NAMESPACE_END
