#include "Common.h"
#include "../Simulation.h"
#include "../obstacles/Obstacle.h"
#include "../obstacles/Sphere.h"

#include <mpi.h>

using namespace cubismup3d;
using namespace cubismup3d::pybindings;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

/* Ensure that we load highest thread level we need. */
struct CUP_MPI_Loader {
  CUP_MPI_Loader() {
    int flag, provided;
    MPI_Initialized(&flag);
    if (!flag) {
      MPI_Init_thread(0, nullptr, MPI_THREAD_MULTIPLE, &provided);
    } else {
      MPI_Query_thread(&provided);
    }
#ifdef CUP_ASYNC_DUMP
    const auto SECURITY = MPI_THREAD_MULTIPLE;
#else
    const auto SECURITY = MPI_THREAD_FUNNELED;
#endif
    if (provided >= SECURITY)
      return;
    if (!flag)
      fprintf(stderr, "Error: MPI implementation does not have the required thread support!\n");
    else
      fprintf(stderr, "Error: MPI does not implement or not initialized with the required thread support!\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
} cup_mpi_loader;


std::shared_ptr<SimulationData> init_SimulationData(
    MPI_Comm comm,
    double CFL,
    std::array<int, 3> cells,
    std::array<Real, 3> uinf)
{
  auto SD = std::make_shared<SimulationData>(comm);
  SD->setCells(cells[0], cells[1], cells[2]);
  SD->CFL = CFL;
  SD->uinf[0] = uinf[0];
  SD->uinf[1] = uinf[1];
  SD->uinf[2] = uinf[2];
  return SD;
}

}  // namespace

PYBIND11_MODULE(cubismup3d, m) {

  m.doc() = "CubismUP3D solver for incompressible Navier-Stokes";

  /* SimulationData */
  SimulationData SD{MPI_COMM_WORLD};  // For default values.
  py::class_<SimulationData, std::shared_ptr<SimulationData>>(m, "SimulationData")
      .def(py::init<MPI_Comm>())
      .def(py::init(&init_SimulationData),
           "comm"_a = MPI_COMM_WORLD,
           "CFL"_a,    // Mandatory.
           "cells"_a,  // Mandatory.
           "uinf"_a = SD.uinf)
      .def_readwrite("CFL", &SimulationData::CFL)
      .def_readwrite("BCx_flag", &SimulationData::BCx_flag, "Boundary condition in x-axis.")
      .def_readwrite("BCy_flag", &SimulationData::BCy_flag, "Boundary condition in y-axis.")
      .def_readwrite("BCz_flag", &SimulationData::BCz_flag, "Boundary condition in z-axis.")
      .def_readwrite("extent", &SimulationData::extent)
      .def_readwrite("uinf", &SimulationData::uinf)
      .def_readwrite("nsteps", &SimulationData::nsteps)
      .def("setCells", &SimulationData::setCells);


  /* Simulation */
  py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation")
      .def(py::init<const SimulationData &>(),
           R"(
               Simulation documentation....
           )")
      .def_readonly("sim", &Simulation::sim, py::return_value_policy::reference)
      .def("run", &Simulation::run)
      .def("add_obstacle", &Simulation_addObstacle);


  bindObstacles(m);
}
