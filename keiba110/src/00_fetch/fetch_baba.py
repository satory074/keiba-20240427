"""
Fetch track condition data from JRA HTML.
Source S3: JRA track condition HTML (https://www.jra.go.jp/keiba/baba/YYYYMMDDⱼⱼ.html)
"""

import datetime as dt
import logging
import pathlib
import time
from typing import Dict, List, Optional, Union

import httpx
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.jra.go.jp/keiba/baba"
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
REQUEST_TIMEOUT = 5  # seconds


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


def save_html(html_content: str, date_str: str, course_code: str, output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save HTML content to file.
    
    Args:
        html_content: HTML content to save
        date_str: Date string in format YYYYMMDD
        course_code: Course code (e.g., 05 for Tokyo)
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{date_str}_{course_code}_baba.html"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved HTML content to {output_file}")
    return output_file


def parse_track_condition(html_content: str, date_str: str, course_code: str) -> Optional[Dict]:
    """
    Parse track condition data from HTML content.
    
    Args:
        html_content: HTML content to parse
        date_str: Date string in format YYYYMMDD
        course_code: Course code (e.g., 05 for Tokyo)
    
    Returns:
        Dictionary with track condition data or None if parsing failed
    """
    try:
        soup = BeautifulSoup(html_content, "lxml")
        
        condition_table = soup.select_one("table.condition")
        if not condition_table:
            logger.warning("Condition table not found in HTML")
            return None
        
        turf_state = None
        turf_row = condition_table.find("th", text=lambda t: t and "芝" in t)
        if turf_row:
            turf_state = turf_row.find_next_sibling("td").text.strip()
        
        dirt_state = None
        dirt_row = condition_table.find("th", text=lambda t: t and "ダート" in t)
        if dirt_row:
            dirt_state = dirt_row.find_next_sibling("td").text.strip()
        
        moisture_front = None
        moisture_row = condition_table.find("th", text=lambda t: t and "含水率" in t)
        if moisture_row:
            moisture_text = moisture_row.find_next_sibling("td").text.strip()
            try:
                moisture_front = float(moisture_text.replace("%", ""))
            except ValueError:
                logger.warning(f"Could not parse moisture value: {moisture_text}")
        
        cushion_value = None
        cushion_row = condition_table.find("th", text=lambda t: t and "クッション値" in t)
        if cushion_row:
            cushion_text = cushion_row.find_next_sibling("td").text.strip()
            try:
                cushion_value = float(cushion_text)
            except ValueError:
                logger.warning(f"Could not parse cushion value: {cushion_text}")
        
        if not turf_state and not dirt_state:
            logger.warning("Both turf and dirt state are missing")
            return None
        
        result = {
            "date": date_str,
            "course": course_code,
            "turf_state": turf_state,
            "dirt_state": dirt_state,
            "moisture_front": moisture_front,
            "cushion": cushion_value,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing track condition: {str(e)}")
        return None


def save_track_condition(data: Dict, output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save track condition data to Parquet file.
    
    Args:
        data: Track condition data as dictionary
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([data])
    
    date_str = data["date"]
    course_code = data["course"]
    output_file = output_dir / f"{date_str}_{course_code}_baba.parquet"
    
    df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved track condition data to {output_file}")
    return output_file


def save_failed_request(date_str: str, course_code: str, error_info: str) -> None:
    """
    Save failed request information for later processing.
    
    Args:
        date_str: Date string in format YYYYMMDD
        course_code: Course code
        error_info: Error information
    """
    failed_dir = pathlib.Path("data/raw/FAILED_RAW")
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    failed_file = failed_dir / f"FAILED_{date_str}_{course_code}_baba_{timestamp}.txt"
    
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"Date: {date_str}\n")
        f.write(f"Course: {course_code}\n")
        f.write(f"Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"Error: {error_info}\n")
    
    logger.warning(f"Saved failed request info to {failed_file}")


def get_baba_url(date_str: str, course_code: str) -> str:
    """
    Generate track condition URL from date and course code.
    
    Args:
        date_str: Date string in format YYYYMMDD
        course_code: Course code (e.g., 05 for Tokyo)
    
    Returns:
        URL to the track condition HTML page
    """
    return f"{BASE_URL}/{date_str}{course_code}.html"


def main(date_str: Optional[str] = None, course_code: str = "05") -> None:
    """
    Main function to fetch and save track condition data.
    
    Args:
        date_str: Date string in format YYYYMMDD (default: today)
        course_code: Course code (default: 05 for Tokyo)
    """
    if date_str is None:
        date_str = dt.datetime.now().strftime("%Y%m%d")
    
    url = get_baba_url(date_str, course_code)
    
    try:
        html_content = fetch_html(url)
        
        if html_content:
            save_html(html_content, date_str, course_code, "data/raw/html")
            
            track_data = parse_track_condition(html_content, date_str, course_code)
            
            if track_data:
                save_track_condition(track_data, "data/raw")
            else:
                error_info = "Failed to parse track condition"
                save_failed_request(date_str, course_code, error_info)
        else:
            error_info = "Failed to fetch HTML"
            save_failed_request(date_str, course_code, error_info)
            
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")
        save_failed_request(date_str, course_code, str(e))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch track condition data from JRA HTML")
    parser.add_argument("--date", default=None, 
                        help="Date in format YYYYMMDD (default: today)")
    parser.add_argument("--course", default="05", 
                        help="Course code (default: 05 for Tokyo)")
    
    args = parser.parse_args()
    main(args.date, args.course)
