//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Cylinder.h"
#include "extra/ObstacleLibrary.h"

#include <Cubism/ArgumentParser.h>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

namespace DCylinderObstacle
{
struct FillBlocks : FillBlocksBase<FillBlocks>
{
  const Real radius, halflength, angle, h, safety = (2+SURFDH)*h;
  const Real cosang = std::cos(angle), sinang = std::sin(angle);
  const double position[3];
  const Real box[3][2] = {
    {(Real)position[0]-radius    -safety, (Real)position[0]+radius    +safety},
    {(Real)position[1]-radius    -safety, (Real)position[1]+radius    +safety},
    {(Real)position[2]-halflength-safety, (Real)position[2]+halflength+safety}
  };

  FillBlocks(const Real r, const Real halfl, const Real ang,
             const Real _h, const double p[3]):
  radius(r), halflength(halfl), angle(ang), h(_h), position{p[0],p[1],p[2]} {}

  inline bool isTouching(const FluidBlock&b, const int buffer_dx=0) const
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
    const Real x = xo - position[0], y = yo - position[1], z = zo - position[2];
    const Real x_rotated =   x * cosang + y * sinang;
    const Real planeDist = std::min( -x_rotated, radius-std::sqrt(x*x+y*y) );
    const Real vertiDist = halflength - std::fabs(z);
    return std::min(planeDist, vertiDist);
  }
};
}

namespace CylinderObstacle
{
struct FillBlocks : FillBlocksBase<FillBlocks>
{
  const Real radius, halflength, h, safety = (2+SURFDH)*h;
  const double position[3];
  const Real box[3][2] = {
    {(Real)position[0]-radius    -safety, (Real)position[0]+radius    +safety},
    {(Real)position[1]-radius    -safety, (Real)position[1]+radius    +safety},
    {(Real)position[2]-halflength-safety, (Real)position[2]+halflength+safety}
  };

  FillBlocks(const Real r, const Real halfl, const Real _h, const double p[3]):
  radius(r), halflength(halfl), h(_h), position{p[0],p[1],p[2]} {}

  inline bool isTouching(const FluidBlock&b, const int buffer_dx=0) const
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
    const Real x = xo - position[0], y = yo - position[1], z = zo - position[2];
    const Real planeDist = radius - std::sqrt(x*x+y*y);
    const Real vertiDist = halflength - std::fabs(z);
    return std::min(planeDist, vertiDist);
  }
};
}

Cylinder::Cylinder(
    SimulationData&s, ArgumentParser &p)
    : Obstacle(s, p), radius(.5 * length),
      halflength(p("-halflength").asDouble(.5 * sim.extent[2]))
{
  section = p("-section").asString("circular");
  accel = p("-accel").asBool(false);
  if(accel) {
    if(not bForcedInSimFrame[0]) {
      printf("Warning: Cylinder was not set to be forced in x-dir, yet the accel pattern is active.\n");
    }
    umax = - p("-xvel").asDouble(0.0);
    tmax = p("-T").asDouble(1.0);
    transVel_imposed[0] = 0;
  }
  _init();
}

Cylinder::Cylinder(
    SimulationData& s,
    ObstacleArguments &args,
    const double radius_,
    const double halflength_)
    : Obstacle(s, args), radius(radius_), halflength(halflength_)
{
  _init();
}

void Cylinder::_init(void)
{
  printf("Created Cylinder with radius %f and halflength %f\n", radius, halflength);

  // D-cyl can float around the domain, but does not support rotation. TODO
  bBlockRotation[0] = true;
  bBlockRotation[1] = true;
  bBlockRotation[2] = true;
}


void Cylinder::create()
{
  const Real h = sim.maxH();
  if(section == "D")
  {
    const DCylinderObstacle::FillBlocks kernel(radius, halflength, _2Dangle,
                                               h, position);
    create_base<DCylinderObstacle::FillBlocks>(kernel);
  }
  else /* else do square section, but figure how to make code smaller */
  {    /* else normal cylinder */
    const CylinderObstacle::FillBlocks kernel(radius, halflength, h, position);
    create_base<CylinderObstacle::FillBlocks>(kernel);
  }
}


void Cylinder::computeVelocities()
{
  if(accel) {
    if(sim.time<tmax) transVel_imposed[0] = umax*sim.time/tmax;
    else transVel_imposed[0] = umax;
  }

  Obstacle::computeVelocities();
}

void Cylinder::finalize()
{
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

CubismUP_3D_NAMESPACE_END
