"""
Unit test for odds fetching.
"""

import json
import unittest
from unittest.mock import patch, MagicMock

import httpx

from src.00_fetch.fetch_ozz import fetch_odds, save_odds


class TestOddsFetch(unittest.TestCase):
    """Test cases for odds fetching functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_odds_data = {
            "data": {
                "odds": [
                    {"horse_id": "2020104325", "horse_name": "テスト馬1", "odds": "5.6"},
                    {"horse_id": "2020104326", "horse_name": "テスト馬2", "odds": "3.2"},
                    {"horse_id": "2020104327", "horse_name": "テスト馬3", "odds": "12.4"},
                ]
            }
        }
        
        self.race_id = "202504260611"

    @patch("httpx.get")
    def test_fetch_odds_success(self, mock_get):
        """Test successful odds fetching."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = self.mock_odds_data
        mock_get.return_value = mock_response
        
        result = fetch_odds(self.race_id)
        
        self.assertEqual(result, self.mock_odds_data)
        
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        self.assertIn(self.race_id, args[0])
        self.assertEqual(kwargs["headers"]["User-Agent"], "keiba110-bot/0.1 (+github)")

    @patch("httpx.get")
    def test_fetch_odds_rate_limit(self, mock_get):
        """Test rate-limited odds fetching."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_get.return_value = mock_response
        
        with patch("time.sleep") as mock_sleep:
            result = fetch_odds(self.race_id)
        
        self.assertIsNone(result)
        
        mock_sleep.assert_called()

    @patch("httpx.get")
    def test_fetch_odds_error(self, mock_get):
        """Test error in odds fetching."""
        mock_get.side_effect = httpx.RequestError("Connection error")
        
        with patch("time.sleep") as mock_sleep:
            result = fetch_odds(self.race_id)
        
        self.assertIsNone(result)
        
        mock_sleep.assert_called()

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open")
    @patch("json.dump")
    def test_save_odds(self, mock_json_dump, mock_open, mock_mkdir):
        """Test saving odds data to file."""
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        output_path = save_odds(self.race_id, self.mock_odds_data, "data/raw")
        
        self.assertIsNotNone(output_path)
        
        mock_mkdir.assert_called_once()
        
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once_with(self.mock_odds_data, mock_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    unittest.main()
