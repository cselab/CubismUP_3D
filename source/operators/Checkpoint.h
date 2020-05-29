//
//  Cubism3D
//  Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Ivica Kicic (kicici@ethz.ch).
//

#ifndef CubismUP_3D_Operators_Checkpoint_h
#define CubismUP_3D_Operators_Checkpoint_h

#include "Operator.h"

#include <functional>
#include <list>

CubismUP_3D_NAMESPACE_BEGIN

/*
 * Checkpoint to which arbitrary code can be attached.
 *
 * This is to allow extending the pipeline with external codes without
 * modification of the CubismUP_3D code.
 *
 * Checkpoints should be assigned a semantical meaning, like before or after
 * some value has been computed. See Simulation.h/.cpp for more details.
 */
class Checkpoint : public Operator
{
  using listener_type = std::function<void(double dt)>;
  using container_type = std::list<listener_type>;
  using iterator = container_type::iterator;

public:
  Checkpoint(SimulationData& s, std::string name);

  std::string getName() override;

  /* Invoke all listeners. */
  void operator()(double dt) override;

  /* Add a listener to the checkpoint. */
  iterator addListener(listener_type listener);

  /* Remove a listener. Passing an already removed listener is an error. */
  void removeListener(iterator it);

private:
  container_type listeners_;
  std::string name_;
};

CubismUP_3D_NAMESPACE_END
#endif
