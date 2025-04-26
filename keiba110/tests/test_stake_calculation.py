"""
Unit test for stake calculation.
"""

import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

import sys
sys.path.append("/home/ubuntu/keiba110")
from src.30_live.live_bet import calculate_stakes


class TestStakeCalculation(unittest.TestCase):
    """Test cases for stake calculation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_data = pd.DataFrame({
            "race_id": ["202504260611"] * 5,
            "horse_id": ["1001", "1002", "1003", "1004", "1005"],
            "horse_name": ["Horse1", "Horse2", "Horse3", "Horse4", "Horse5"],
            "odds_win": [2.5, 5.0, 10.0, 20.0, 50.0],
            "odds_close": [2.0, 6.0, 9.0, 25.0, 40.0],
            "p_blend": [0.5, 0.25, 0.15, 0.05, 0.02],
        })
        
        self.bankroll = 10000

    @patch("pathlib.Path.glob")
    @patch("builtins.open")
    @patch("json.load")
    def test_calculate_stakes(self, mock_json_load, mock_open, mock_glob):
        """Test stake calculation."""
        mock_glob.return_value = ["data/raw/202504260611_odds_202504261200.json"]
        
        mock_json_load.return_value = {
            "data": {
                "odds": [
                    {"horse_id": "1001", "odds": "2.0"},
                    {"horse_id": "1002", "odds": "6.0"},
                    {"horse_id": "1003", "odds": "9.0"},
                    {"horse_id": "1004", "odds": "25.0"},
                    {"horse_id": "1005", "odds": "40.0"},
                ]
            }
        }
        
        result = calculate_stakes(self.test_data, self.bankroll)
        
        self.assertEqual(len(result), 5)
        
        np.testing.assert_almost_equal(result["EV"].values, 
                                      self.test_data["p_blend"] * self.test_data["odds_win"] * 0.8)
        
        np.testing.assert_almost_equal(result["ROI"].values, result["EV"].values - 1)
        
        np.testing.assert_almost_equal(result["CLV"].values, 
                                      self.test_data["odds_win"] - self.test_data["odds_close"])
        
        
        self.assertEqual(result.loc[result["horse_id"] == "1003", "stake"].values[0], 100)
        
        self.assertEqual(result.loc[result["horse_id"] == "1001", "stake"].values[0], 0)
        self.assertEqual(result.loc[result["horse_id"] == "1002", "stake"].values[0], 0)
        self.assertEqual(result.loc[result["horse_id"] == "1004", "stake"].values[0], 0)
        self.assertEqual(result.loc[result["horse_id"] == "1005", "stake"].values[0], 0)

    def test_calculate_stakes_high_roi(self):
        """Test stake calculation with high ROI."""
        test_data_high_roi = self.test_data.copy()
        test_data_high_roi["p_blend"] = [0.8, 0.25, 0.15, 0.05, 0.02]  # Horse1 has high probability
        
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = []
            
            result = calculate_stakes(test_data_high_roi, self.bankroll)
        
        self.assertEqual(result.loc[result["horse_id"] == "1001", "stake"].values[0], 0)

    def test_calculate_stakes_empty_data(self):
        """Test stake calculation with empty data."""
        result = calculate_stakes(pd.DataFrame(), self.bankroll)
        
        self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
