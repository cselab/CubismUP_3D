//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch) in May 2018.
//

#include "BufferedLogger.h"

#include <fstream>
#include <unordered_map>

namespace cubismup3d {

BufferedLogger logger;

struct BufferedLoggerImpl {
  struct Stream {
    std::stringstream stream;
    int requests_since_last_flush = 0;

    // GN: otherwise icpc complains
    Stream() = default;
    Stream(Stream &&) = default;
    Stream(const Stream& c) : requests_since_last_flush(c.requests_since_last_flush)
    {
      stream << c.stream.rdbuf();
    }
  };
  typedef std::unordered_map<std::string, Stream> container_type;
  container_type files;

  /*
   * Flush a single stream and reset the counter.
   */
  void flush(container_type::value_type &p) {
    std::ofstream savestream;
    savestream.open(p.first, std::ios::app | std::ios::out);
    savestream << p.second.stream.rdbuf();
    savestream.close();
    p.second.requests_since_last_flush = 0;
  }

  std::stringstream& get_stream(const std::string &filename) {
    auto it = files.find(filename);
    if (it != files.end()) {
      if (++it->second.requests_since_last_flush == BufferedLogger::AUTO_FLUSH_COUNT)
        flush(*it);
      return it->second.stream;
    } else {
      // With request_since_last_flush == 0,
      // the first flush will have AUTO_FLUSH_COUNT frames.
      auto new_it = files.emplace(filename, Stream()).first;
      return new_it->second.stream;
    }
  }
};

BufferedLogger::BufferedLogger() : impl(new BufferedLoggerImpl) { }

BufferedLogger::~BufferedLogger() {
  flush();
  delete impl;
}

std::stringstream& BufferedLogger::get_stream(const std::string &filename) {
  return impl->get_stream(filename);
}

void BufferedLogger::flush(void) {
  for (auto &pair : impl->files)
    impl->flush(pair);
}

}  // namespace cubismup3d
