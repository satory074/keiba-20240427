"""
Fetch annual moisture data from JRA PDF.
Source S5: JRA annual moisture PDF (https://www.jra.go.jp/keiba/baba/gansui/YYYY.html)
"""

import datetime as dt
import logging
import os
import pathlib
import re
import time
from typing import Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
from bs4 import BeautifulSoup
import pdfplumber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.jra.go.jp/keiba/baba/gansui"
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
REQUEST_TIMEOUT = 10  # seconds

COURSE_NAMES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


def fetch_html(url: str) -> Optional[str]:
    """
    Fetch HTML content with retry logic.
    
    Args:
        url: URL to fetch
    
    Returns:
        HTML content as string or None if all retries failed
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching HTML from {url}, attempt {attempt}/{MAX_RETRIES}")
            response = httpx.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.text
            else:
                logger.warning(f"HTTP error: {response.status_code}")
            
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                
        except httpx.RequestError as e:
            logger.error(f"Error fetching HTML: {str(e)}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
    
    logger.error(f"All retries failed for {url}")
    return None


def save_html(html_content: str, year: str, output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save HTML content to file.
    
    Args:
        html_content: HTML content to save
        year: Year string in format YYYY
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{year}_gansui.html"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved HTML content to {output_file}")
    return output_file


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
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    logger.info(f"Successfully downloaded PDF to {output_path}")
                    return True
                else:
                    logger.warning(f"Downloaded PDF is empty: {output_path}")
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


def extract_pdf_links(html_content: str) -> List[Tuple[str, str]]:
    """
    Extract PDF links from HTML content.
    
    Args:
        html_content: HTML content to parse
    
    Returns:
        List of tuples (PDF URL, PDF filename)
    """
    soup = BeautifulSoup(html_content, "lxml")
    links = []
    
    for a_tag in soup.find_all("a", href=re.compile(r"\.pdf$")):
        href = a_tag.get("href")
        if href:
            filename = os.path.basename(href)
            links.append((href, filename))
    
    return links


def extract_moisture_data(pdf_path: Union[str, pathlib.Path]) -> List[Dict]:
    """
    Extract moisture data from PDF file.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        List of dictionaries with moisture data
    """
    result = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                
                tables = page.extract_tables()
                
                if not tables:
                    logger.warning(f"No tables found in {pdf_path}, page {page.page_number}")
                    continue
                
                for table in tables:
                    if not table or len(table) <= 1:
                        continue
                    
                    header = table[0]
                    data_rows = table[1:]
                    
                    for row in data_rows:
                        if not row or all(cell is None or cell.strip() == "" for cell in row):
                            continue
                        
                        date_str = None
                        course_code = None
                        
                        for i, cell in enumerate(row):
                            if cell and isinstance(cell, str):
                                date_match = re.search(r"(\d{4})/(\d{1,2})/(\d{1,2})", cell)
                                if date_match:
                                    year, month, day = date_match.groups()
                                    date_str = f"{year}{month:0>2}{day:0>2}"
                                
                                for code, name in COURSE_NAMES.items():
                                    if name in cell:
                                        course_code = code
                                        break
                        
                        turf_moist = None
                        dirt_moist = None
                        
                        for i, cell in enumerate(row):
                            if cell and isinstance(cell, str):
                                moist_match = re.search(r"(\d+\.\d+)%", cell)
                                if moist_match:
                                    moist_value = float(moist_match.group(1))
                                    
                                    if "芝" in cell:
                                        turf_moist = moist_value
                                    elif "ダート" in cell:
                                        dirt_moist = moist_value
                        
                        if date_str and course_code:
                            entry = {
                                "date": date_str,
                                "course": course_code,
                                "turf_moist": turf_moist,
                                "dirt_moist": dirt_moist,
                            }
                            result.append(entry)
        
        return result
        
    except Exception as e:
        logger.error(f"Error extracting moisture data: {str(e)}")
        return []


def save_moisture_data(data: List[Dict], output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save moisture data to Parquet file.
    
    Args:
        data: List of dictionaries with moisture data
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(data)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    output_file = output_dir / f"gansui_{timestamp}.parquet"
    
    df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved moisture data to {output_file}")
    return output_file


def save_failed_request(year: str, error_info: str) -> None:
    """
    Save failed request information for later processing.
    
    Args:
        year: Year string in format YYYY
        error_info: Error information
    """
    failed_dir = pathlib.Path("data/raw/FAILED_RAW")
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    failed_file = failed_dir / f"FAILED_{year}_gansui_{timestamp}.txt"
    
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"Year: {year}\n")
        f.write(f"Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"Error: {error_info}\n")
    
    logger.warning(f"Saved failed request info to {failed_file}")


def get_gansui_url(year: str) -> str:
    """
    Generate annual moisture URL from year.
    
    Args:
        year: Year string in format YYYY
    
    Returns:
        URL to the annual moisture HTML page
    """
    return f"{BASE_URL}/{year}.html"


def main(year: Optional[str] = None) -> None:
    """
    Main function to fetch and save annual moisture data.
    
    Args:
        year: Year string in format YYYY (default: current year)
    """
    if year is None:
        year = dt.datetime.now().strftime("%Y")
    
    url = get_gansui_url(year)
    
    try:
        html_content = fetch_html(url)
        
        if html_content:
            save_html(html_content, year, "data/raw/html")
            
            pdf_links = extract_pdf_links(html_content)
            
            if pdf_links:
                all_moisture_data = []
                
                for pdf_url, pdf_filename in pdf_links:
                    if not pdf_url.startswith("http"):
                        if pdf_url.startswith("/"):
                            pdf_url = f"https://www.jra.go.jp{pdf_url}"
                        else:
                            pdf_url = f"{BASE_URL}/{pdf_url}"
                    
                    pdf_dir = pathlib.Path("data/raw/pdf")
                    pdf_dir.mkdir(parents=True, exist_ok=True)
                    
                    pdf_path = pdf_dir / pdf_filename
                    
                    if download_pdf(pdf_url, pdf_path):
                        moisture_data = extract_moisture_data(pdf_path)
                        
                        if moisture_data:
                            all_moisture_data.extend(moisture_data)
                        else:
                            logger.warning(f"No moisture data extracted from {pdf_filename}")
                    else:
                        logger.warning(f"Failed to download PDF: {pdf_url}")
                
                if all_moisture_data:
                    save_moisture_data(all_moisture_data, "data/raw")
                else:
                    error_info = "No moisture data extracted from any PDF"
                    save_failed_request(year, error_info)
            else:
                error_info = "No PDF links found in HTML"
                save_failed_request(year, error_info)
        else:
            error_info = "Failed to fetch HTML"
            save_failed_request(year, error_info)
            
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        save_failed_request(year, str(e))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch annual moisture data from JRA PDF")
    parser.add_argument("--year", default=None, 
                        help="Year in format YYYY (default: current year)")
    
    args = parser.parse_args()
    main(args.year)
