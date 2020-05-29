//
//  Cubism3D
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland.
//  Distributed under the terms of the MIT license.
//
//  Created by Christian Conti.
//

#ifndef CubismUP_3D_Timer_h
#define CubismUP_3D_Timer_h

#include <chrono>

CubismUP_3D_NAMESPACE_BEGIN

class Timer
{
    std::chrono::time_point<std::chrono::high_resolution_clock> t_start, t_end;

public:

	void start()
	{
		t_start = std::chrono::high_resolution_clock::now();
	}

	double stop()
	{
		t_end = std::chrono::high_resolution_clock::now();

		return std::chrono::duration<double>(t_end-t_start).count();
	}
};

CubismUP_3D_NAMESPACE_END
#endif // CubismUP_3D_Timer_h
