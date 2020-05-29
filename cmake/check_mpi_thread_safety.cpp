/*
 * Used by CMakeLists.txt to check whether MPI_THREAD_MULTIPLE is supported.
 *
 * If not, asynchronous dumping is disabled and synchronous is used instead.
 */
#include <mpi.h>

int main(int argc, char **argv) {
    int provided;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided < MPI_THREAD_MULTIPLE)
        MPI_Abort(MPI_COMM_WORLD, 1);  // :(

    MPI_Finalize();

    return 0;  // :)
}
