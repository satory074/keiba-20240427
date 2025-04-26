"""
Fetch odds data from netkeiba API.
Source S1: netkeiba odds API (https://race.netkeiba.com/api/api_get_jra_odds.html)
"""

import datetime as dt
import json
import logging
import pathlib
import time
from typing import Dict, Optional, Union

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"
HEADERS = {"User-Agent": "keiba110-bot/0.1 (+github)"}
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
REQUEST_TIMEOUT = 5  # seconds
RATE_LIMIT_SLEEP = 120  # 2 minutes between requests
BAN_SLEEP = 600  # 10 minutes sleep when banned (HTTP 429)


def fetch_odds(race_id: str, odds_type: int = 1) -> Optional[Dict]:
    """
    Fetch odds data from netkeiba API with retry logic.
    
    Args:
        race_id: Race ID in format YYYYMMDDCCRRD (e.g., 202504260611)
        odds_type: Type of odds (1=win, 2=place, etc.)
    
    Returns:
        JSON response as dictionary or None if all retries failed
    """
    url = f"{BASE_URL}?type={odds_type}&locale=ja&race_id={race_id}"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching odds for race {race_id}, attempt {attempt}/{MAX_RETRIES}")
            response = httpx.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429). Sleeping for {BAN_SLEEP} seconds")
                time.sleep(BAN_SLEEP)
            else:
                logger.warning(f"HTTP error: {response.status_code}")
                
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching odds: {str(e)}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
    
    logger.error(f"All retries failed for race {race_id}")
    return None


def save_odds(race_id: str, data: Dict, output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save odds data to JSON file.
    
    Args:
        race_id: Race ID
        data: Odds data as dictionary
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{race_id}_odds_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved odds data to {output_file}")
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
    failed_file = failed_dir / f"FAILED_{race_id}_odds_{timestamp}.txt"
    
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"Race ID: {race_id}\n")
        f.write(f"Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"Error: {error_info}\n")
    
    logger.warning(f"Saved failed request info to {failed_file}")


def main(race_id: str = "202504260611") -> None:
    """
    Main function to fetch and save odds data in a loop.
    
    Args:
        race_id: Race ID to fetch odds for
    """
    output_dir = pathlib.Path("data/raw")
    
    while dt.datetime.now().hour < 17:
        try:
            odds_data = fetch_odds(race_id)
            
            if odds_data:
                save_odds(race_id, odds_data, output_dir)
            else:
                error_info = "All retries failed"
                save_failed_request(race_id, error_info)
                
            logger.info(f"Sleeping for {RATE_LIMIT_SLEEP} seconds")
            time.sleep(RATE_LIMIT_SLEEP)
            
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            save_failed_request(race_id, str(e))
            time.sleep(RATE_LIMIT_SLEEP)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch odds data from netkeiba API")
    parser.add_argument("race_id", nargs="?", default="202504260611", 
                        help="Race ID in format YYYYMMDDCCRRD (default: 202504260611)")
    
    args = parser.parse_args()
    main(args.race_id)
