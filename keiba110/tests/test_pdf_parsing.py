"""
Unit test for PDF parsing.
"""

import os
import pathlib
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
import PyPDF2

from src.00_fetch.fetch_entries import extract_entries, validate_pdf


class TestPDFParsing(unittest.TestCase):
    """Test cases for PDF parsing functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = pathlib.Path("tests/test_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_pdf = self.test_dir / "test_race_card.pdf"
        
        self.mock_pdf_content = b"%PDF-1.5\n...test content..."
        
        if not self.test_pdf.exists():
            with open(self.test_pdf, "wb") as f:
                f.write(self.mock_pdf_content)

    def tearDown(self):
        """Tear down test fixtures."""
        if self.test_pdf.exists():
            os.remove(self.test_pdf)

    @patch("PyPDF2.PdfReader")
    def test_validate_pdf(self, mock_pdf_reader):
        """Test PDF validation."""
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [MagicMock(), MagicMock()]  # Two pages
        mock_reader_instance.pages[0].extract_text.return_value = "Test text"
        
        mock_pdf_reader.return_value = mock_reader_instance
        
        result = validate_pdf(self.test_pdf)
        self.assertTrue(result)
        
        mock_reader_instance.pages = []
        result = validate_pdf(self.test_pdf)
        self.assertFalse(result)
        
        mock_reader_instance.pages = [MagicMock()]
        mock_reader_instance.pages[0].extract_text.return_value = ""
        result = validate_pdf(self.test_pdf)
        self.assertFalse(result)

    @patch("tabula.read_pdf")
    def test_extract_entries(self, mock_read_pdf):
        """Test extraction of entries from PDF."""
        mock_df = pd.DataFrame({
            0: [1, 2, 3, 4, 5],
            1: ["101", "102", "103", "104", "105"],
            2: ["Horse1", "Horse2", "Horse3", "Horse4", "Horse5"],
            3: ["牡3", "牝4", "セ5", "牡6", "牝3"],
            4: [480, 460, 490, 500, 470],
            5: ["Jockey1", "Jockey2", "Jockey3", "Jockey4", "Jockey5"],
            6: ["Trainer1", "Trainer2", "Trainer3", "Trainer4", "Trainer5"],
            7: ["Owner1", "Owner2", "Owner3", "Owner4", "Owner5"],
        })
        
        mock_read_pdf.return_value = [mock_df]
        
        result = extract_entries(self.test_pdf)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertEqual(list(result[2]), ["Horse1", "Horse2", "Horse3", "Horse4", "Horse5"])
        
        mock_read_pdf.return_value = []
        result = extract_entries(self.test_pdf)
        self.assertIsNone(result)
        
        mock_read_pdf.return_value = [mock_df.iloc[:3]]
        result = extract_entries(self.test_pdf)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
