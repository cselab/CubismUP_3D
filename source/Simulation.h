//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Written by Guido Novati (novatig@ethz.ch).
//

#ifndef CubismUP_3D_Simulation_h
#define CubismUP_3D_Simulation_h

#include "SimulationData.h"

#include <memory>

// Forward declarations.
namespace cubism { class ArgumentParser; }

CubismUP_3D_NAMESPACE_BEGIN

class Checkpoint;
class Obstacle;

class Simulation
{
  //#ifdef _USE_ZLIB_
  //  SerializerIO_WaveletCompression_MPI_SimpleBlocking<FluidGridMPI, ChiStreamer> waveletdumper_grid;
  //#endif

public:

  SimulationData sim;
  Checkpoint *checkpointPreObstacles = nullptr;
  Checkpoint *checkpointPostVelocity = nullptr;

  void reset();
  void _init(bool restart = false);
  void _serialize(const std::string append = std::string());
  void _deserialize();

  void _argumentsSanityCheck();
  void setupOperators();
  void setupGrid(cubism::ArgumentParser *parser_ptr = nullptr);
  void _ic();
  void _icFromH5(std::string h5File);

 public:
  Simulation(const SimulationData &);
  Simulation(MPI_Comm mpicomm);
  Simulation(MPI_Comm mpicomm, cubism::ArgumentParser &parser);

  virtual ~Simulation() = default;

  virtual void run();

  // void addObstacle(IF3D_ObstacleOperator *obstacle);
  // void removeObstacle(IF3D_ObstacleOperator *obstacle);

  /* Get reference to the obstacle container. */
  const std::vector<std::shared_ptr<Obstacle>> &getObstacleVector() const;

  /* Calculate maximum allowed time step, including CFL and ramp-up. */
  double calcMaxTimestep();

  /*
   * Perform one timestep of the simulation.
   *
   * Returns true if the simulation is finished.
   */
  bool timestep(double dt);
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Simulation_h
