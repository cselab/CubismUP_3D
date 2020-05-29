//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Written by Ivica Kicic (kicici@ethz.ch).
//

#include "Plate.h"
#include "extra/ObstacleLibrary.h"

#include <Cubism/ArgumentParser.h>

CubismUP_3D_NAMESPACE_BEGIN
using namespace cubism;

//static constexpr Real EPSILON = std::numeric_limits<Real>::epsilon();

static void _normalize(
    double * const x,
    double * const y,
    double * const z) {
  const double norm = std::sqrt(*x * *x + *y * *y + *z * *z);
  assert(norm > 1e-9);
  const double inv = 1.0 / norm;
  *x = inv * *x;
  *y = inv * *y;
  *z = inv * *z;
}

static void _normalized_cross(
    const double ax,
    const double ay,
    const double az,
    const double bx,
    const double by,
    const double bz,
    double * const cx,
    double * const cy,
    double * const cz) {
  const double x = ay * bz - az * by;
  const double y = az * bx - ax * bz;
  const double z = ax * by - ay * bx;
  const double norm = std::sqrt(x * x + y * y + z * z);
  assert(norm > 1e-9);
  const double inv = 1.0 / norm;
  *cx = inv * x;
  *cy = inv * y;
  *cz = inv * z;
}


////////////////////////////////////////////////////////////
// PLATE FILL BLOCKS
////////////////////////////////////////////////////////////
namespace
{
  struct PlateFillBlocks : FillBlocksBase<PlateFillBlocks>
  {
    const Real cx, cy, cz;      // Center.
    const Real nx, ny, nz;      // Normal. NORMALIZED.
    const Real ax, ay, az;      // A-side vector. NORMALIZED.
    const Real bx, by, bz;      // A-side vector. NORMALIZED.
    const Real half_a;          // Half-size in A direction.
    const Real half_b;          // Half-size in B direction.
    const Real half_thickness;  // Half-thickess. Edges are rounded.

    Real aabb[3][2];            // Axis-aligned bounding box.

    PlateFillBlocks(
        Real cx, Real cy, Real cz,
        Real nx, Real ny, Real nz,
        Real ax, Real ay, Real az,
        Real bx, Real by, Real bz,
        Real half_a,
        Real half_b,
        Real half_thickness, Real h);

    // Required by FillBlocksBase.
    bool isTouching(const FluidBlock&b) const;
    Real signedDistance(Real x, Real y, Real z) const;
  };
}  // Anonymous namespace.


PlateFillBlocks::PlateFillBlocks(
    const Real _cx, const Real _cy, const Real _cz,
    const Real _nx, const Real _ny, const Real _nz,
    const Real _ax, const Real _ay, const Real _az,
    const Real _bx, const Real _by, const Real _bz,
    const Real _half_a,
    const Real _half_b,
    const Real _half_thickness, const Real h)
  : cx(_cx), cy(_cy), cz(_cz),
    nx(_nx), ny(_ny), nz(_nz),
    ax(_ax), ay(_ay), az(_az),
    bx(_bx), by(_by), bz(_bz),
    half_a(_half_a),
    half_b(_half_b),
    half_thickness(_half_thickness)
{
  using std::fabs;

  // Assert normalized.
  assert(fabs(nx * nx + ny * ny + nz * nz - 1) < (Real)1e-9);
  assert(fabs(ax * ax + ay * ay + az * az - 1) < (Real)1e-9);
  assert(fabs(bx * bx + by * by + bz * bz - 1) < (Real)1e-9);

  // Assert n, a and b are mutually orthogonal.
  assert(fabs(nx * ax + ny * ay + nz * az) < (Real)1e-9);
  assert(fabs(nx * bx + ny * by + nz * bz) < (Real)1e-9);
  assert(fabs(ax * bx + ay * by + az * bz) < (Real)1e-9);

  const double skin = (2 + SURFDH) * h; 
  const double tx = skin + fabs(ax * half_a) + fabs(bx * half_b) + fabs(nx * half_thickness);
  const double ty = skin + fabs(ay * half_a) + fabs(by * half_b) + fabs(ny * half_thickness);
  const double tz = skin + fabs(az * half_a) + fabs(bz * half_b) + fabs(nz * half_thickness);

  aabb[0][0] = cx - tx;
  aabb[0][1] = cx + tx;
  aabb[1][0] = cy - ty;
  aabb[1][1] = cy + ty;
  aabb[2][0] = cz - tz;
  aabb[2][1] = cz + tz;
}

bool PlateFillBlocks::isTouching(const FluidBlock&b) const
{
  return aabb[0][0] <= b.max_pos[0] && aabb[0][1] >= b.min_pos[0]
      && aabb[1][0] <= b.max_pos[1] && aabb[1][1] >= b.min_pos[1]
      && aabb[2][0] <= b.max_pos[2] && aabb[2][1] >= b.min_pos[2];
}

Real PlateFillBlocks::signedDistance(
    const Real x,
    const Real y,
    const Real z) const
{
  // Move plane to the center.
  const Real dx = x - cx;
  const Real dy = y - cy;
  const Real dz = z - cz;
  const Real dotn = dx * nx + dy * ny + dz * nz;

  // Project (x, y, z) to the centered plane.
  const Real px = dx - dotn * nx;
  const Real py = dy - dotn * ny;
  const Real pz = dz - dotn * nz;

  // Project into directions a and b.
  const Real dota = px * ax + py * ay + pz * az;
  const Real dotb = px * bx + py * by + pz * bz;

  // Distance to the rectangle edges in the plane coordinate system.
  const Real a = std::fabs(dota) - half_a;
  const Real b = std::fabs(dotb) - half_b;
  const Real n = std::fabs(dotn) - half_thickness;

  if (a <= 0 && b <= 0 && n <= 0) {
    // Inside, return a positive number.
    return -std::min(n, std::min(a, b));
  } else {
    // Outside, return a negative number.
    const Real a0 = std::max((Real)0, a);
    const Real b0 = std::max((Real)0, b);
    const Real n0 = std::max((Real)0, n);
    return -std::sqrt(a0 * a0 + b0 * b0 + n0 * n0);
  }

  // ROUNDED EDGES.
  // return half_thickness - std::sqrt(dotn * dotn + a0 * a0 + b0 * b0);
}

////////////////////////////////////////////////////////////
// PLATE OBSTACLE OPERATOR
////////////////////////////////////////////////////////////

Plate::Plate(SimulationData & s, ArgumentParser &p) : Obstacle(s, p)
{
  p.set_strict_mode();
  half_a = (Real)0.5 * p("-a").asDouble();
  half_b = (Real)0.5 * p("-b").asDouble();
  half_thickness = (Real)0.5 * p("-thickness").asDouble();
  p.unset_strict_mode();

  bool has_alpha = p.check("-alpha");
  if (has_alpha) {
    _from_alpha(M_PI / 180.0 * p("-alpha").asDouble());
  } else {
    p.set_strict_mode();
    nx = p("-nx").asDouble();
    ny = p("-ny").asDouble();
    nz = p("-nz").asDouble();
    ax = p("-ax").asDouble();
    ay = p("-ay").asDouble();
    az = p("-az").asDouble();
    p.unset_strict_mode();
  }

  _init();
}

Plate::Plate(
    SimulationData & s,
    ObstacleArguments &args,
    const double a,
    const double b,
    const double thickness,
    const double alpha)
  : Obstacle(s, args),
    half_a(.5 * a),
    half_b(.5 * b),
    half_thickness(.5 * thickness)
{
  _from_alpha(alpha);
  _init();
}

Plate::Plate(
    SimulationData & s,
    ObstacleArguments &args,
    const double a,
    const double b,
    const double thickness,
    const double _nx, const double _ny, const double _nz,
    const double _ax, const double _ay, const double _az)
  : Obstacle(s, args),
    nx(_nx), ny(_ny), nz(_nz),
    ax(_ax), ay(_ay), az(_az),
    half_a(.5 * a),
    half_b(.5 * b),
    half_thickness(.5 * thickness)
{
  _init();
}

void Plate::_from_alpha(const double alpha)
{
  nx = std::cos(alpha);
  ny = std::sin(alpha);
  nz = 0;
  ax = -std::sin(alpha);
  ay = std::cos(alpha);
  az = 0;
}

void Plate::_init(void)
{
  _normalize(&nx, &ny, &nz);
  _normalized_cross(nx, ny, nz, ax, ay, az, &bx, &by, &bz);
  _normalized_cross(bx, by, bz, nx, ny, nz, &ax, &ay, &az);

  // Plateq can float around the domain, but does not support rotation. TODO
  bBlockRotation[0] = true;
  bBlockRotation[1] = true;
  bBlockRotation[2] = true;
}


void Plate::create()
{
  const Real h = sim.maxH();
  const PlateFillBlocks K(
      position[0], position[1], position[2],
      nx, ny, nz,
      ax, ay, az,
      bx, by, bz,
      half_a, half_b, half_thickness, h);

  create_base<PlateFillBlocks>(K);
}

void Plate::finalize()
{
  // this method allows any computation that requires the char function
  // to be computed. E.g. compute the effective center of mass or removing
  // momenta from udef
}

CubismUP_3D_NAMESPACE_END
