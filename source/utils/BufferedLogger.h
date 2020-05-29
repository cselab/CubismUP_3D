//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

#ifndef CubismUP_3D_utils_BufferedLogger_h
#define CubismUP_3D_utils_BufferedLogger_h

#include <sstream>

namespace cubismup3d {

struct BufferedLoggerImpl;
/*
 * Buffered file logging with automatic flush.
 *
 * A stream is flushed periodically.
 * (Such that the user doesn't have to manually call flush.)
 *
 * If killing intentionally simulation, don't forget to flush the logger!
 */
class BufferedLogger {
  BufferedLoggerImpl * const impl;

public:
  static constexpr int AUTO_FLUSH_COUNT = 100;

  BufferedLogger();
  BufferedLogger(const BufferedLogger &) = delete;
  BufferedLogger(BufferedLogger &&) = delete;
  ~BufferedLogger();

  /* Flush all streams. */
  void flush(void);

  /*
   * Get or create a string for a given file name.
   *
   * The stream is automatically flushed if accessed
   * many times since last flush.
   */
  std::stringstream& get_stream(const std::string &filename);
};

extern BufferedLogger logger;  // Declared in BufferedLogger.cpp.

}  // namespace cubismup3d

#endif // CubismUP_3D_utils_BufferedLogger_h
