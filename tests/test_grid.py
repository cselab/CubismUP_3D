#!/usr/bin/env python3

import unittest

from utils import TestCaseEx
import cubismup3d as C

class TestGrid(TestCaseEx):
    def test_cells_are_required(self):
        with self.assertRaises(TypeError):
            C.Simulation()  # Error: Cells not specified.

    def test_cell_multiple_of_block_size(self):
        """Non-multiple of block size should raise an error."""
        with self.assertRaises(ValueError):
            C.Simulation(cells=[128, 128, 129])
        with self.assertRaises(ValueError):
            C.Simulation(cells=[128, 129, 128])
        with self.assertRaises(ValueError):
            C.Simulation(cells=[129, 128, 128])

        # This one does not crash...
        c = C.Simulation(cells=[64, 32, 32])



if __name__ == '__main__':
    unittest.main()
