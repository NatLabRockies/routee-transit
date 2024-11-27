# Tests for GTFS processing module
import unittest
from unittest import TestCase
from ..GTFS_processing import GTFS_processing


def test_createDFs(self):
    expected_result = "89590"
    actual_result = len(GTFS_processing.create_dfs("saltlake")[0])
    self.assertEqual(expected_result, actual_result)

def test_select_bus_shapes(self):
    expected_result = 199
    actual_result = len(GTFS_processing.select_bus_shapes())
    self.assertEqual(expected_result, actual_result)

if __name__ == '__main__':
    unittest.main()