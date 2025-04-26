"""
Fetch weather forecast data from JMA JSON.
Source S4: JMA weather forecast JSON (https://www.jma.go.jp/bosai/forecast/data/forecast/AREA.json)
"""

import datetime as dt
import json
import logging
import pathlib
import time
from typing import Dict, List, Optional, Union

import httpx
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = "https://www.jma.go.jp/bosai/forecast/data/forecast"
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
REQUEST_TIMEOUT = 5  # seconds

AREA_CODES = {
    "Tokyo": "130000",     # 東京
    "Nakayama": "120000",  # 中山 (千葉)
    "Kyoto": "260000",     # 京都
    "Hanshin": "280000",   # 阪神 (兵庫)
    "Chukyo": "230000",    # 中京 (愛知)
    "Sapporo": "016000",   # 札幌
    "Hakodate": "017000",  # 函館
    "Fukushima": "070000", # 福島
    "Niigata": "150000",   # 新潟
    "Kokura": "400000",    # 小倉 (福岡)
}

COURSE_TO_AREA = {
    "01": "Sapporo",
    "02": "Hakodate",
    "03": "Fukushima",
    "04": "Niigata",
    "05": "Tokyo",
    "06": "Nakayama",
    "07": "Chukyo",
    "08": "Kyoto",
    "09": "Hanshin",
    "10": "Kokura",
}


def fetch_weather(area_code: str) -> Optional[Dict]:
    """
    Fetch weather forecast data with retry logic.
    
    Args:
        area_code: Area code (e.g., 130000 for Tokyo)
    
    Returns:
        JSON response as dictionary or None if all retries failed
    """
    url = f"{BASE_URL}/{area_code}.json"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching weather for area {area_code}, attempt {attempt}/{MAX_RETRIES}")
            response = httpx.get(url, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"HTTP error: {response.status_code}")
            
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
                
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching weather: {str(e)}")
            if attempt < MAX_RETRIES:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.info(f"Retrying in {sleep_time:.1f} seconds")
                time.sleep(sleep_time)
    
    logger.error(f"All retries failed for area {area_code}")
    return None


def save_weather(area_name: str, data: Dict, output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save weather data to JSON file.
    
    Args:
        area_name: Area name (e.g., Tokyo)
        data: Weather data as dictionary
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{area_name}_weather_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved weather data to {output_file}")
    return output_file


def parse_weather(data: Dict, area_name: str) -> List[Dict]:
    """
    Parse weather forecast data from JSON response.
    
    Args:
        data: Weather data as dictionary
        area_name: Area name (e.g., Tokyo)
    
    Returns:
        List of dictionaries with parsed weather data
    """
    result = []
    
    try:
        forecasts = data[0]["timeSeries"][0]["timeDefines"]
        weather_codes = data[0]["timeSeries"][0]["areas"][0]["weatherCodes"]
        weather_texts = data[0]["timeSeries"][0]["areas"][0]["weathers"]
        
        temps_max = None
        temps_min = None
        
        if len(data[0]["timeSeries"]) > 2:
            temp_series = data[0]["timeSeries"][2]
            temps_max = temp_series["areas"][0]["temps"]
            
            temps_max = [int(t) if t not in ["-", ""] else None for t in temps_max]
            
            if "tempsMin" in temp_series["areas"][0]:
                temps_min = temp_series["areas"][0]["tempsMin"]
                temps_min = [int(t) if t not in ["-", ""] else None for t in temps_min]
        
        winds = None
        if len(data[0]["timeSeries"]) > 1:
            wind_series = data[0]["timeSeries"][1]
            winds = wind_series["areas"][0]["winds"]
        
        for i, forecast_time in enumerate(forecasts):
            entry = {
                "ts": forecast_time,
                "area": area_name,
                "wx_code": weather_codes[i] if i < len(weather_codes) else None,
                "wx": weather_texts[i] if i < len(weather_texts) else None,
                "temp_max": temps_max[i] if temps_max and i < len(temps_max) else None,
                "temp_min": temps_min[i] if temps_min and i < len(temps_min) else None,
                "wind": winds[i] if winds and i < len(winds) else None,
            }
            result.append(entry)
        
        return result
        
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing weather data: {str(e)}")
        return []


def save_weather_parquet(weather_data: List[Dict], output_dir: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Save parsed weather data to Parquet file.
    
    Args:
        weather_data: List of dictionaries with weather data
        output_dir: Directory to save the file
    
    Returns:
        Path to the saved file
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(weather_data)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    output_file = output_dir / f"weather_{timestamp}.parquet"
    
    df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved weather data to {output_file}")
    return output_file


def save_failed_request(area_name: str, error_info: str) -> None:
    """
    Save failed request information for later processing.
    
    Args:
        area_name: Area name that failed
        error_info: Error information
    """
    failed_dir = pathlib.Path("data/raw/FAILED_RAW")
    failed_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    failed_file = failed_dir / f"FAILED_{area_name}_weather_{timestamp}.txt"
    
    with open(failed_file, "w", encoding="utf-8") as f:
        f.write(f"Area: {area_name}\n")
        f.write(f"Timestamp: {dt.datetime.now().isoformat()}\n")
        f.write(f"Error: {error_info}\n")
    
    logger.warning(f"Saved failed request info to {failed_file}")


def main(area_names: Optional[List[str]] = None) -> None:
    """
    Main function to fetch and save weather data for specified areas.
    
    Args:
        area_names: List of area names to fetch weather for (default: all areas)
    """
    if area_names is None:
        area_names = list(AREA_CODES.keys())
    
    all_weather_data = []
    
    for area_name in area_names:
        if area_name not in AREA_CODES:
            logger.warning(f"Unknown area: {area_name}")
            continue
        
        area_code = AREA_CODES[area_name]
        
        try:
            weather_data = fetch_weather(area_code)
            
            if weather_data:
                save_weather(area_name, weather_data, "data/raw/json")
                
                parsed_data = parse_weather(weather_data, area_name)
                
                if parsed_data:
                    all_weather_data.extend(parsed_data)
                else:
                    error_info = "Failed to parse weather data"
                    save_failed_request(area_name, error_info)
            else:
                error_info = "Failed to fetch weather data"
                save_failed_request(area_name, error_info)
                
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            save_failed_request(area_name, str(e))
    
    if all_weather_data:
        save_weather_parquet(all_weather_data, "data/raw")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch weather forecast data from JMA JSON")
    parser.add_argument("--areas", nargs="+", default=None, 
                        help="Area names to fetch weather for (default: all areas)")
    
    args = parser.parse_args()
    main(args.areas)
