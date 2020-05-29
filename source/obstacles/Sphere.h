//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Sphere_h
#define CubismUP_3D_Sphere_h

#include "Obstacle.h"

CubismUP_3D_NAMESPACE_BEGIN


struct SphereArguments
{
  const double radius;
  double umax = 0;
  double tmax = 1;
  //special case: startup with unif accel to umax in tmax, and then decel to 0
  bool accel_decel = false;
  bool bHemi = false;

  SphereArguments(double R) : radius(R) {}
  SphereArguments(double R, double _umax, double _tmax, bool _ad, bool _bHemi)
      : radius(R), umax(_umax), tmax(_tmax), accel_decel(_ad), bHemi(_bHemi) { }
};

struct ObstacleAndSphereArguments : ObstacleArguments, SphereArguments {
  ObstacleAndSphereArguments(ObstacleArguments o, SphereArguments s)
      : ObstacleArguments(std::move(o)), SphereArguments(std::move(s)) { }
};


class Sphere : public Obstacle, private SphereArguments
{
public:
  Sphere(SimulationData&s,cubism::ArgumentParser&p);
  Sphere(SimulationData&s,ObstacleArguments&args,double R);
  Sphere(SimulationData&s,ObstacleArguments&args,double R, double umax, double tmax);
  Sphere(SimulationData&s, const ObstacleAndSphereArguments &args);

  void create() override;
  void finalize() override;
  void computeVelocities() override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Sphere_h
