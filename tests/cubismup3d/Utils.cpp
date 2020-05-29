#include "Utils.h"
#include <mpi.h>

namespace cubismup3d {
namespace tests {

void init_mpi(int *argc, char ***argv) {
  int provided;
  #ifdef CUP_ASYNC_DUMP
    const auto SECURITY = MPI_THREAD_MULTIPLE;
  #else
    const auto SECURITY = MPI_THREAD_FUNNELED;
  #endif
  MPI_Init_thread(argc, argv, SECURITY, &provided);
  if (provided < SECURITY) {
    fprintf(stderr, "ERROR: MPI implementation does not have required thread support\n");
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void finalize_mpi(void) {
  MPI_Finalize();
}

}  // testt
}  // cubismup3d
