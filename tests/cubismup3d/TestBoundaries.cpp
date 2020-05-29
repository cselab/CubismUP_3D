#include "Utils.h"
#include "../../source/Simulation.h"
#include "../../source/operators/CellwiseOperator.h"

using namespace cubism;
using namespace cubismup3d;

static constexpr int CELLS_X = 128;
static constexpr int CELLS_Y = 32;
static constexpr int CELLS_Z = 32;

/* Expected value of a given cell. */
double getElementValue(int abs_ix, int abs_iy, int abs_iz)
{
  return abs_ix + 1000 * abs_iy + 1000000 * abs_iz;
}

/* Test one stencil on periodic boundaries with a ready simulation object. */
void _testPeriodicBoundaries(Simulation &S, StencilInfo stencil, int dx, int dy, int dz)
{
  // Reset the grid to some initial vlaue.
  applyKernel(S.sim, [](CellInfo info, FluidElement &e) {
    e.u = getElementValue(info.get_abs_ix(), info.get_abs_iy(), info.get_abs_iz());
  });

  // Shift everything in the direction (dx, dy, dz);
  applyStencilKernel(
    S.sim,
    stencil,
    [dx, dy, dz](const StencilKernelLab &lab, CellInfo info, FluidElement &out) {
      out.tmpU = lab(-dx, -dy, -dz).u;
    }
  );

  // Check if everything is correct now.
  applyKernel(S.sim, [dx, dy, dz](CellInfo info, FluidElement &e) {
    int aix = (info.get_abs_ix() - dx + CELLS_X) % CELLS_X;
    int aiy = (info.get_abs_iy() - dy + CELLS_Y) % CELLS_Y;
    int aiz = (info.get_abs_iz() - dz + CELLS_Z) % CELLS_Z;
    double expected = getElementValue(aix, aiy, aiz);
    if (expected != e.tmpU) {
      fprintf(stderr, "Value (%d, %d, %d) in the block (%d %d %d) is %lf instead of %lf\n",
              info.ix, info.iy, info.iz,
              info.block_info.index[0],
              info.block_info.index[1],
              info.block_info.index[2],
              e.tmpU, expected);
      fprintf(stderr, "Failed at shift (%d, %d, %d)\n", dx, dy, dz);;
      exit(1);
    }
  });
}

/* Test stencils on periodic boundaries. */
bool testPeriodicBoundaries()
{
  // Prepare simulation data and the simulation object.
  auto prepareSimulationData = []() {
    SimulationData SD{MPI_COMM_WORLD};
    SD.CFL = 0.1;
    SD.BCx_flag = periodic;  // <--- Periodic boundaries.
    SD.BCy_flag = periodic;
    SD.BCz_flag = periodic;
    SD.setCells(CELLS_X, CELLS_Y, CELLS_Z);
    return SD;
  };
  Simulation S{prepareSimulationData()};

  // Try out 3 different stencils.
  _testPeriodicBoundaries(S, StencilInfo(-1, -1, -1, 2, 2, 2, false, {{FE_U}}), -1, 0, 0);
  _testPeriodicBoundaries(S, StencilInfo(-1, -1, -1, 2, 2, 2, false, {{FE_U}}), +1, 0, 0);
  _testPeriodicBoundaries(S, StencilInfo(-2, -2, -2, 3, 3, 3, true,  {{FE_U}}), -1, +1, +2);

  return true;
}

int main(int argc, char **argv)
{
  tests::init_mpi(&argc, &argv);

  CUP_RUN_TEST(testPeriodicBoundaries);

  tests::finalize_mpi();
}
