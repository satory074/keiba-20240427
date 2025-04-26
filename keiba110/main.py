"""
Keiba110 - Horse racing prediction and betting system.
Main entry point for the application.
"""

import argparse
import datetime as dt
import logging
import pathlib
import subprocess
import sys
from typing import List, Optional

from src.00_fetch.fetch_ozz import main as fetch_odds
from src.00_fetch.fetch_entries import main as fetch_entries
from src.00_fetch.fetch_baba import main as fetch_baba
from src.00_fetch.fetch_weather import main as fetch_weather
from src.10_feature.build_features import main as build_features_main
from src.20_model.train_stack import main as train_model_main
from src.30_live.live_bet import main as live_bet_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_directories() -> None:
    """
    Set up required directories.
    """
    dirs = [
        "data/raw",
        "data/raw/FAILED_RAW",
        "data/raw/html",
        "data/raw/json",
        "data/raw/pdf",
        "data/processed",
        "data/model",
        "data/bets",
    ]
    
    for dir_path in dirs:
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Directories set up successfully")


def fetch_data(race_id: str) -> None:
    """
    Fetch data for a specific race.
    
    Args:
        race_id: Race ID in format YYYYMMDDCCRRD
    """
    
    date_str = race_id[:8]  # YYYYMMDD
    course_code = race_id[8:10]  # CC
    
    logger.info(f"Fetching odds for race {race_id}")
    fetch_odds(race_id)
    
    logger.info(f"Fetching entries for race {race_id}")
    fetch_entries(race_id)
    
    logger.info(f"Fetching track condition for date {date_str} and course {course_code}")
    fetch_baba(date_str, course_code)
    
    logger.info("Fetching weather data")
    fetch_weather()
    
    logger.info("Data fetching complete")


def build_features() -> None:
    """
    Build features from raw data.
    """
    
    logger.info("Building features")
    build_features_main()
    logger.info("Feature building complete")


def train_model() -> None:
    """
    Train the prediction model.
    """
    
    logger.info("Training model")
    train_model_main()
    logger.info("Model training complete")


def live_betting(race_id: str) -> None:
    """
    Run live betting for a specific race.
    
    Args:
        race_id: Race ID in format YYYYMMDDCCRRD
    """
    
    logger.info(f"Running live betting for race {race_id}")
    live_bet_main(race_id)
    logger.info("Live betting complete")


def run_dashboard() -> None:
    """
    Run the Streamlit dashboard.
    """
    
    logger.info("Starting dashboard")
    subprocess.run(["streamlit", "run", "src/40_dashboard/dashboard.py"])


def main() -> None:
    """
    Main function to run the application.
    """
    parser = argparse.ArgumentParser(description="Keiba110 - Horse racing prediction and betting system")
    parser.add_argument("race_id", nargs="?", default=None, 
                        help="Race ID in format YYYYMMDDCCRRD")
    parser.add_argument("--fetch", action="store_true", 
                        help="Fetch data for the specified race")
    parser.add_argument("--features", action="store_true", 
                        help="Build features from raw data")
    parser.add_argument("--train", action="store_true", 
                        help="Train the prediction model")
    parser.add_argument("--bet", action="store_true", 
                        help="Run live betting for the specified race")
    parser.add_argument("--dashboard", action="store_true", 
                        help="Run the Streamlit dashboard")
    parser.add_argument("--all", action="store_true", 
                        help="Run the complete pipeline")
    
    args = parser.parse_args()
    
    setup_directories()
    
    if (args.fetch or args.bet or args.all) and not args.race_id:
        logger.error("Race ID is required for fetch, bet, or all operations")
        parser.print_help()
        sys.exit(1)
    
    if args.all:
        fetch_data(args.race_id)
        build_features()
        train_model()
        live_betting(args.race_id)
        run_dashboard()
    else:
        if args.fetch:
            fetch_data(args.race_id)
        
        if args.features:
            build_features()
        
        if args.train:
            train_model()
        
        if args.bet:
            live_betting(args.race_id)
        
        if args.dashboard:
            run_dashboard()
        
        if not (args.fetch or args.features or args.train or args.bet or args.dashboard):
            parser.print_help()


if __name__ == "__main__":
    main()
