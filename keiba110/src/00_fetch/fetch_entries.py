"""
Fetch race entries from JRA PDF files.
Source S2: JRA race card PDFs (https://www.jra.go.jp/keiba/rpdf/YYYYMM/DD/ⱼⱼⱼ.pdf)
"""

import datetime as dt
import logging
import os
import pathlib
import time
from typing import Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
import PyPDF2
import tabula

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.jra.go.jp/keiba/rpdf"
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
REQUEST_TIMEOUT = 10  # seconds
MIN_EXPECTED_PAGES = 1
MIN_EXPECTED_HORSES = 5  # Minimum number of horses expected in a race


def download_pdf(url: str, output_path: Union[str, pathlib.Path]) -> bool:
    """
    Download PDF file with retry logic.
    
    Args:
        url: URL of the PDF file
        output_path: Path to save the PDF file
    
    Returns:
        True if download was successful, False otherwise
    """
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Downloading PDF from {url}, attempt {attempt}/{MAX_RETRIES}")
            response = httpx.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(response.content)
                
                if validate_pdf(output_path):
                    logger.info(f"Successfully downloaded and validated PDF to {output_path}")
                    return True
                else:
                    logger.warning(f"Downloaded PDF failed validation: {output_path}")
                    if attempt < MAX_RETRIES:
                        os.remove(output_path)  # Remove invalid PDF
            else:
                logger.warning(f"HTTP error: {response.status_code}")
            
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                
        except httpx.RequestError as e:
            logger.error(f"Error downloading PDF: {str(e)}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
    
    logger.error(f"All retries failed for {url}")
    return False


def validate_pdf(pdf_path: Union[str, pathlib.Path]) -> bool:
    """
    Validate downloaded PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        True if PDF is valid, False otherwise
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            
            if num_pages < MIN_EXPECTED_PAGES:
                logger.warning(f"PDF has fewer pages than expected: {num_pages} < {MIN_EXPECTED_PAGES}")
                return False
            
            text = pdf_reader.pages[0].extract_text()
            if not text:
                logger.warning("PDF first page has no extractable text")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"Error validating PDF: {str(e)}")
        return False


def extract_entries(pdf_path: Union[str, pathlib.Path]) -> Optional[pd.DataFrame]:
    """
    Extract race entries from PDF file using tabula.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        DataFrame with race entries or None if extraction failed
    """
    try:
        tables = tabula.read_pdf(
            pdf_path,
            pages="1",
            area=[108, 28, 790, 566],
            columns=[28, 70, 138, 278, 330, 382, 450, 512],
            pandas_options={"header": None},
        )
        
        if not tables or len(tables) == 0:
            logger.warning(f"No tables extracted from {pdf_path}")
            return None
        
        df = tables[0]
        
        if len(df) < MIN_EXPECTED_HORSES:
            logger.warning(f"Fewer horses than expected: {len(df)} < {MIN_EXPECTED_HORSES}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error extracting entries from PDF: {str(e)}")
        return None


def save_entries(df: pd.DataFrame, race_id: str, output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save extracted entries to Parquet file.
    
    Args:
        df: DataFrame with race entries
        race_id: Race ID
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df["race_id"] = race_id
    
    column_names = ["draw", "horse_id", "horse_name", "sex_age", "weight", "jockey", "trainer", "owner", "race_id"]
    if len(df.columns) >= len(column_names):
        df.columns = column_names + [f"col{i}" for i in range(len(df.columns) - len(column_names))]
    
    output_file = output_dir / f"{race_id}_entries.parquet"
    df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved entries data to {output_file}")
    return output_file


def save_failed_request(race_id: str, error_info: str) -> None:
    """
    Save failed request information for later processing.
    
    Args:
        race_id: Race ID that failed
        error_info: Error information
    """
    failed_dir = pathlib.Path("data/raw/FAILED_RAW")
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    failed_file = failed_dir / f"FAILED_{race_id}_entries_{timestamp}.txt"
    
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"Race ID: {race_id}\n")
        f.write(f"Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"Error: {error_info}\n")
    
    logger.warning(f"Saved failed request info to {failed_file}")


def get_pdf_url(race_id: str) -> str:
    """
    Generate PDF URL from race ID.
    
    Args:
        race_id: Race ID in format YYYYMMDDCCRRD (e.g., 202504260611)
    
    Returns:
        URL to the PDF file
    """
    year_month = race_id[:6]  # YYYYMM
    day = race_id[6:8]  # DD
    course_code = race_id[8:10]  # CC
    race_num = race_id[12:]  # R
    
    return f"{BASE_URL}/{year_month}/{day}/{course_code}{race_num}.pdf"


def main(race_id: str = "202504260611") -> None:
    """
    Main function to download PDF and extract entries.
    
    Args:
        race_id: Race ID to fetch entries for
    """
    pdf_url = get_pdf_url(race_id)
    pdf_dir = pathlib.Path("data/raw/pdf")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_path = pdf_dir / f"{race_id}.pdf"
    
    try:
        if download_pdf(pdf_url, pdf_path):
            df = extract_entries(pdf_path)
            
            if df is not None:
                save_entries(df, race_id, "data/raw")
            else:
                error_info = "Failed to extract entries from PDF"
                save_failed_request(race_id, error_info)
        else:
            error_info = "Failed to download PDF"
            save_failed_request(race_id, error_info)
            
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        save_failed_request(race_id, str(e))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch race entries from JRA PDF files")
    parser.add_argument("race_id", nargs="?", default="202504260611", 
                        help="Race ID in format YYYYMMDDCCRRD (default: 202504260611)")
    
    args = parser.parse_args()
    main(args.race_id)
