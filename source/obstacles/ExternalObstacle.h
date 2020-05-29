//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

#ifndef CubismUP_3D_ExternalObstacle_h
#define CubismUP_3D_ExternalObstacle_h

// XXX: DEPRECATED, use ObstacleFromShape instead. Remove in 2020.

/*
 * This obstacle can be used to insert obstacles whose shape and velocity is
 * defined by an external code. Intended to be used when CubismUP_3D used as a
 * library.
 */
#include "Obstacle.h"

#include <functional>

CubismUP_3D_NAMESPACE_BEGIN

/*
 * Callbacks and other information for `IF3D_ExternalObstacleOperator`.
 *
 * This structure enables the user to define a custom obstacle.
 */
struct ExternalObstacleArguments
{
  typedef std::array<Real, 3> Point;
  typedef std::array<Real, 3> Velocity;

  /*
   * Check if given box is touching (intersecting) the object.
   *
   * False positives are allowed.
   */
  std::function<bool(Point low, Point high)> isTouchingFn;

  /*
   * Returns the signed distance to the object boundary.
   *
   * Positive values are to be returned for points inside the object,
   * negative for points outside of the object. Must be precise only close to
   * the obstacle surface.
   */
  std::function<Real(Point)> signedDistanceFn;

  /* Returns the local object velocity at the given location. */
  std::function<Velocity(Point)> velocityFn;

  /* Returns the center-of-mass velocity of the object. */
  std::function<Point()> comVelocityFn;

  /* Returns the lambda factor (opacity), given the current time. */
  std::function<double(double)> lambdaFactorFn;
};

struct ObstacleAndExternalArguments : ObstacleArguments, ExternalObstacleArguments
{
  ObstacleAndExternalArguments() = default;
  ObstacleAndExternalArguments(ObstacleArguments o, ExternalObstacleArguments e);
};

class ExternalObstacle : public Obstacle, private ExternalObstacleArguments
{
public:
  ExternalObstacle(SimulationData &s, const ObstacleAndExternalArguments &);

  void computeVelocities() override;
  void create() override;
  void finalize() override;
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_ExternalObstacle_h
