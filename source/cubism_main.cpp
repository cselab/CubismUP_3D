//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Simulation.h"

#include <Cubism/ArgumentParser.h>

#include <cmath>
#include <iostream>
#include <unistd.h>

using std::cout;

int cubism_main (const MPI_Comm app_comm, int argc, char **argv)
{
  int rank;
  MPI_Comm_rank(app_comm, &rank);

  cubism::ArgumentParser parser(argc,argv);
  parser.set_strict_mode();

  int supported_threads;
  MPI_Query_thread(&supported_threads);
  if (supported_threads < MPI_THREAD_FUNNELED) {
    printf("ERROR: The MPI implementation does not have required thread support\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  char hostname[1024];
  hostname[1023] = '\0';
  gethostname(hostname, 1023);
  printf("Rank %d is on host %s\n", rank, hostname);

  if (rank==0) {
    cout << "====================================================================================================================\n";
    cout << "\t\tCubism UP 3D (velocity-pressure 3D incompressible Navier-Stokes solver)\n";
    cout << "====================================================================================================================\n";
  }

  cubismup3d::Simulation sim(app_comm, parser);
  sim.run();

  return 0;
}
