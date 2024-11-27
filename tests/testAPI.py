#module to test API that fetches GTFS data
import unittest
from unittest import TestCase
from ..GTFS_processing import GTFS_data


def test_API(self):
    key = GTFS_data.get_api_key
    expected_result = 200
    actual_result = GTFS_data.check_api_status(key)
    self.assertEqual(expected_result, actual_result)

def test_getGTFS_zip(self):
    key = GTFS_data.get_api_key
    expected_result = 3
    actual_result = GTFS_data.getGTFS("rhode island", key)
    self.assertEqual(expected_result, actual_result)

if __name__ == '__main__':
    unittest.main()
    