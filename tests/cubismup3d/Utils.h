#ifndef CUBISMUP3D_TESTS_UTILS_H
#define CUBISMUP3D_TESTS_UTILS_H

#include <cstdio>

namespace cubismup3d {
namespace tests {

void init_mpi(int *argc, char ***argv);
void finalize_mpi(void);

#define CUP_RUN_TEST(test) do { \
      if(!(test)()) { \
        fprintf(stderr, "Failed on test \"%s\"!\n", #test); \
        exit(1); \
      } \
    } while (0)
#define CUP_CHECK(condition, ...) do { \
      if (!(condition)) {  \
        fprintf(stderr, __VA_ARGS__); \
        exit(1); \
      } \
    } while (0);


}  // testt
}  // cubismup3d

#endif
