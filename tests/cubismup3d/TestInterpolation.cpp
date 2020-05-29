#include "Utils.h"
#include "../../source/Simulation.h"
#include "../../source/operators/CellwiseOperator.h"
#include "../../source/operators/LinearInterpolation.h"

using namespace cubism;
using namespace cubismup3d;

static double getValue(std::array<Real, 3> p) {
  // Must be linear (we are testing linear interpolation).
  return 100.0 * p[0] + 2343 * p[1] + 123. * p[2];
}

static bool testLinearInterpolation() {
  constexpr int NUM_POINTS = 1000;
  constexpr Real extent = 100.0;

  // Prepare simulation data and the simulation object.
  auto prepareSimulationData = []() {
    SimulationData SD{MPI_COMM_WORLD};
    SD.CFL = 0.1;
    SD.BCx_flag = periodic;  // <--- Periodic boundaries.
    SD.BCy_flag = periodic;
    SD.BCz_flag = periodic;
    SD.extent[0] = extent;
    SD.setCells(64, 128, 32);
    return SD;
  };
  Simulation S{prepareSimulationData()};

  // Reset the grid to some initial vlaue.
  cubismup3d::applyKernel(S.sim, [](cubismup3d::CellInfo info, FluidElement &e) {
    e.u = getValue(info.get_pos());
  });

  // Generate random points. Avoid boundaries.
  std::vector<std::array<Real, 3>> points;
  points.reserve(NUM_POINTS);
  for (int i = 0; i < NUM_POINTS; ++i) {
    Real x = S.sim.extent[0] * (0.1 + 0.8 / RAND_MAX * rand());
    Real y = S.sim.extent[1] * (0.1 + 0.8 / RAND_MAX * rand());
    Real z = S.sim.extent[2] * (0.1 + 0.8 / RAND_MAX * rand());
    points.push_back({x, y, z});
  }
  std::vector<double> result(NUM_POINTS);

  // Interpolate.
  cubismup3d::linearCellCenteredInterpolation(
      S.sim,
      points,
      [](const FluidElement &e) { return e.u; },      // What to interpolate.
      [&result](int k, double v) { result[k] = v; },  // Where to store the result.
      std::vector<int>({FE_U}));                      // Components for stencil.

  // Check if result correct.
  for (int i = 0; i < NUM_POINTS; ++i) {
    double expected = getValue(points[i]);
    if (std::fabs(expected - result[i]) / expected > 1e-9) {
      fprintf(stderr, "Expected %lf, got %lf. Point is (%lf, %lf %lf).\n",
              expected, result[i], points[i][0], points[i][1], points[i][2]);
      return false;
    }
  }

  return true;
}

int main(int argc, char **argv) {
  tests::init_mpi(&argc, &argv);

  CUP_RUN_TEST(testLinearInterpolation);

  tests::finalize_mpi();
}
