//
//  Cubism3D
//  Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch).
//

#include "Checkpoint.h"

CubismUP_3D_NAMESPACE_BEGIN

Checkpoint::Checkpoint(SimulationData &s, std::string name)
    : Operator(s), name_(std::move(name)) { }

std::string Checkpoint::getName() {
  return "Checkpoint["  + name_ + "]";
}

void Checkpoint::operator()(const double dt) {
  for (auto &listener : listeners_)
    listener(dt);
}

Checkpoint::iterator
Checkpoint::addListener(Checkpoint::listener_type listener) {
  listeners_.push_back(std::move(listener));
  return --listeners_.end();
}

void Checkpoint::removeListener(Checkpoint::iterator it) {
  listeners_.erase(it);
}

CubismUP_3D_NAMESPACE_END
