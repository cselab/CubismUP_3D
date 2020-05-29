#!/usr/bin/env python3

from utils import TestCaseEx
import sys
import cubismup3d as cup
import unittest

class TestObstacles(TestCaseEx):
    def test_sphere(self):
        SD = cup.SimulationData(cells=[64, 64, 64], CFL=0.1, uinf=[0.1, 0.0, 0.0])
        SD.nsteps = 10

        S = cup.Simulation(SD)
        S.add_obstacle(cup.Sphere(position=[0.3, 0.4, 0.5], radius=0.1))

        # What do I test here?
        S.run()


if __name__ == '__main__':
    unittest.main()
