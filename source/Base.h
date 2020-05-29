//
//  Cubism3D
//  Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch).
//

#ifndef CubismUP_3D_Base_h
#define CubismUP_3D_Base_h

/*
 * Macros and definition used by other header files.
 */

// Are we compiling from CubismUP_3D's makefile?
#ifndef CUP_NO_MACROS_HEADER
// No, it's either CMake or external code. Load compile-time settings from this header file.
#include "../build/include/CubismUP3DMacros.h"
#endif

#ifndef CubismUP_3D_NAMESPACE_BEGIN
#define CubismUP_3D_NAMESPACE_BEGIN namespace cubismup3d {
#endif

#ifndef CubismUP_3D_NAMESPACE_END
#define CubismUP_3D_NAMESPACE_END   }  // namespace cubismup3d
#endif

#ifndef CUP_ALIGNMENT
#define CUP_ALIGNMENT 64
#endif
#define CUBISM_ALIGNMENT CUP_ALIGNMENT


CubismUP_3D_NAMESPACE_BEGIN

#ifndef CUP_SINGLE_PRECISION
typedef double Real;
#else
typedef float Real;
#endif

#ifndef CUP_HDF5_DOUBLE_PRECISION
typedef float DumpReal;
#else
typedef double DumpReal;
#endif

CubismUP_3D_NAMESPACE_END

#endif  // CubismUP_3D_Base_h
