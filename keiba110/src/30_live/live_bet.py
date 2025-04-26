"""
Live betting system.
ROI/CLV check, 3-tier staking (300 / 100 / skip), dummy PAT submission.
"""

import datetime as dt
import json
import logging
import math
import pathlib
import pickle
import time
from typing import Dict, List, Optional, Tuple, Union

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = pathlib.Path("data/processed")
MODEL_DIR = pathlib.Path("data/model")
RAW_DATA_DIR = pathlib.Path("data/raw")
ODDS_API_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"
HEADERS = {"User-Agent": "keiba110-bot/0.1 (+github)"}
REQUEST_TIMEOUT = 5  # seconds
RATE_LIMIT_SLEEP = 120  # 2 minutes between requests
BAN_SLEEP = 600  # 10 minutes sleep when banned (HTTP 429)
BANKROLL = 10000  # Initial bankroll in yen


def load_models() -> Tuple:
    """
    Load trained models from disk.
    
    Returns:
        Tuple of (lgb_model, xgb_model, logit_model, lgb_calibrator, xgb_calibrator, logit_calibrator, blender, feature_names)
    """
    try:
        lgb_model = lgb.Booster(model_file=str(MODEL_DIR / "lgb_model.txt"))
        
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(MODEL_DIR / "xgb_model.json"))
        
        with open(MODEL_DIR / "logit_model.pkl", "rb") as f:
            logit_model = pickle.load(f)
        
        with open(MODEL_DIR / "lgb_calibrator.pkl", "rb") as f:
            lgb_calibrator = pickle.load(f)
        
        with open(MODEL_DIR / "xgb_calibrator.pkl", "rb") as f:
            xgb_calibrator = pickle.load(f)
        
        with open(MODEL_DIR / "logit_calibrator.pkl", "rb") as f:
            logit_calibrator = pickle.load(f)
        
        with open(MODEL_DIR / "blender.pkl", "rb") as f:
            blender = pickle.load(f)
        
        with open(MODEL_DIR / "feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        
        logger.info("Successfully loaded all models")
        
        return (
            lgb_model,
            xgb_model,
            logit_model,
            lgb_calibrator,
            xgb_calibrator,
            logit_calibrator,
            blender,
            feature_names,
        )
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise


def fetch_odds(race_id: str, odds_type: int = 1) -> Optional[Dict]:
    """
    Fetch odds data from netkeiba API.
    
    Args:
        race_id: Race ID in format YYYYMMDDCCRRD (e.g., 202504260611)
        odds_type: Type of odds (1=win, 2=place, etc.)
    
    Returns:
        JSON response as dictionary or None if request failed
    """
    url = f"{ODDS_API_URL}?type={odds_type}&locale=ja&race_id={race_id}"
    
    try:
        logger.info(f"Fetching odds for race {race_id}")
        response = httpx.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            logger.warning(f"Rate limited (429). Sleeping for {BAN_SLEEP} seconds")
            time.sleep(BAN_SLEEP)
        else:
            logger.warning(f"HTTP error: {response.status_code}")
            
    except (httpx.RequestError, json.JSONDecodeError) as e:
        logger.error(f"Error fetching odds: {str(e)}")
    
    return None


def save_odds(race_id: str, data: Dict) -> pathlib.Path:
    """
    Save odds data to JSON file.
    
    Args:
        race_id: Race ID
        data: Odds data as dictionary
    
    Returns:
        Path to the saved file
    """
    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M")
    output_dir = RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"{race_id}_odds_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved odds data to {output_file}")
    return output_file


def load_features(race_id: str) -> pd.DataFrame:
    """
    Load features for a specific race.
    
    Args:
        race_id: Race ID
    
    Returns:
        DataFrame with features for the specified race
    """
    features_file = PROCESSED_DATA_DIR / "features.parquet"
    
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(features_file)
        
        race_df = df[df["race_id"] == race_id]
        
        if race_df.empty:
            logger.warning(f"No features found for race {race_id}")
        else:
            logger.info(f"Loaded features for race {race_id} with shape: {race_df.shape}")
        
        return race_df
        
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        return pd.DataFrame()


def predict_probabilities(df: pd.DataFrame, feature_names: List[str],
                         lgb_model: lgb.Booster, xgb_model: xgb.Booster, logit_model: LogisticRegression,
                         lgb_calibrator: IsotonicRegression, xgb_calibrator: IsotonicRegression, logit_calibrator: IsotonicRegression,
                         blender: LogisticRegression) -> pd.DataFrame:
    """
    Predict win probabilities using the stacked ensemble model.
    
    Args:
        df: DataFrame with features
        feature_names: List of feature names used by the models
        lgb_model: LightGBM model
        xgb_model: XGBoost model
        logit_model: Logistic Regression model
        lgb_calibrator: LightGBM calibrator
        xgb_calibrator: XGBoost calibrator
        logit_calibrator: Logistic Regression calibrator
        blender: Stacking blender model
    
    Returns:
        DataFrame with predicted probabilities
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    missing_features = [f for f in feature_names if f not in result.columns]
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        
        for feature in missing_features:
            result[feature] = 0
    
    X = result[feature_names].fillna(0)
    
    lgb_preds = lgb_model.predict(X)
    
    xgb_dmatrix = xgb.DMatrix(X)
    xgb_preds = xgb_model.predict(xgb_dmatrix)
    
    logit_X = logit_model.scaler.transform(X)
    logit_preds = logit_model.predict_proba(logit_X)[:, 1]
    
    lgb_cal = lgb_calibrator.predict(lgb_preds)
    xgb_cal = xgb_calibrator.predict(xgb_preds)
    logit_cal = logit_calibrator.predict(logit_preds)
    
    meta_features = np.column_stack([lgb_cal, xgb_cal, logit_cal])
    
    blend_preds = blender.predict_proba(meta_features)[:, 1]
    
    result["p_lgb"] = lgb_cal
    result["p_xgb"] = xgb_cal
    result["p_logit"] = logit_cal
    result["p_blend"] = blend_preds
    
    return result


def calculate_stakes(df: pd.DataFrame, bankroll: float = BANKROLL) -> pd.DataFrame:
    """
    Calculate stakes based on predicted probabilities and odds.
    
    Args:
        df: DataFrame with predicted probabilities and odds
        bankroll: Current bankroll in yen
    
    Returns:
        DataFrame with calculated stakes
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    latest_odds_files = sorted(list(RAW_DATA_DIR.glob(f"{result['race_id'].iloc[0]}_odds_*.json")))
    
    if not latest_odds_files:
        logger.warning("No historical odds files found for CLV calculation")
        result["odds_close"] = result["odds_win"]
    else:
        earliest_odds_file = latest_odds_files[0]
        
        try:
            with open(earliest_odds_file, "r", encoding="utf-8") as f:
                earliest_odds_data = json.load(f)
            
            earliest_odds = {}
            for horse in earliest_odds_data.get("data", {}).get("odds", []):
                horse_id = str(horse.get("horse_id", ""))
                odds_win = float(horse.get("odds", 0))
                earliest_odds[horse_id] = odds_win
            
            result["odds_close"] = result["horse_id"].map(earliest_odds)
            
        except Exception as e:
            logger.error(f"Error loading earliest odds: {str(e)}")
            result["odds_close"] = result["odds_win"]
    
    result["EV"] = result["p_blend"] * result["odds_win"] * 0.8  # 払戻率補正
    result["ROI"] = result["EV"] - 1
    
    result["CLV"] = result["odds_win"] - result["odds_close"]
    
    result["edge"] = result["ROI"]
    
    result["k"] = 0.25  # ¼ Kelly
    
    for i, row in result.iterrows():
        edge = row["edge"]
        k = 0.25
        
        if edge <= 0:
            continue
        
        while math.exp(-2 * edge**2 * bankroll / 1.2) > 0.05:
            k /= 2
        
        result.at[i, "k"] = k
    
    result["stake"] = 0  # Default to no bet
    
    for i, row in result.iterrows():
        if row["ROI"] >= 0.30 and row["CLV"] > 0:
            kelly_stake = round(bankroll * row["k"] * row["edge"] / (row["odds_win"] - 1) / 100) * 100
            result.at[i, "stake"] = max(300, kelly_stake)
        elif row["ROI"] >= 0.10 and row["CLV"] > 0:
            result.at[i, "stake"] = 100
    
    return result


def submit_bet(horse_id: str, race_id: str, stake: float, odds: float) -> bool:
    """
    Submit bet to PAT (dummy implementation).
    
    Args:
        horse_id: Horse ID
        race_id: Race ID
        stake: Stake amount in yen
        odds: Current odds
    
    Returns:
        True if bet was successfully submitted, False otherwise
    """
    logger.info(f"[DUMMY] Submitting bet: Race {race_id}, Horse {horse_id}, Stake {stake} yen, Odds {odds}")
    
    success = True
    
    if success:
        logger.info("[DUMMY] Bet successfully submitted")
    else:
        logger.error("[DUMMY] Failed to submit bet")
    
    return success


def save_bet_record(race_id: str, horse_id: str, stake: float, odds: float, timestamp: str) -> None:
    """
    Save bet record to file.
    
    Args:
        race_id: Race ID
        horse_id: Horse ID
        stake: Stake amount in yen
        odds: Current odds
        timestamp: Timestamp of the bet
    """
    bet_dir = pathlib.Path("data/bets")
    bet_dir.mkdir(parents=True, exist_ok=True)
    
    bet_file = bet_dir / "bets.csv"
    
    if not bet_file.exists():
        with open(bet_file, "w", encoding="utf-8") as f:
            f.write("timestamp,race_id,horse_id,stake,odds\n")
    
    with open(bet_file, "a", encoding="utf-8") as f:
        f.write(f"{timestamp},{race_id},{horse_id},{stake},{odds}\n")
    
    logger.info(f"Saved bet record to {bet_file}")


def main(race_id: str) -> None:
    """
    Main function to run live betting.
    
    Args:
        race_id: Race ID to bet on
    """
    try:
        (
            lgb_model,
            xgb_model,
            logit_model,
            lgb_calibrator,
            xgb_calibrator,
            logit_calibrator,
            blender,
            feature_names,
        ) = load_models()
        
        while dt.datetime.now().hour < 17:
            odds_data = fetch_odds(race_id)
            
            if odds_data:
                save_odds(race_id, odds_data)
                
                features_df = load_features(race_id)
                
                if not features_df.empty:
                    for horse in odds_data.get("data", {}).get("odds", []):
                        horse_id = str(horse.get("horse_id", ""))
                        odds_win = float(horse.get("odds", 0))
                        
                        mask = features_df["horse_id"] == horse_id
                        if mask.any():
                            features_df.loc[mask, "odds_win"] = odds_win
                    
                    pred_df = predict_probabilities(
                        features_df,
                        feature_names,
                        lgb_model,
                        xgb_model,
                        logit_model,
                        lgb_calibrator,
                        xgb_calibrator,
                        logit_calibrator,
                        blender,
                    )
                    
                    stakes_df = calculate_stakes(pred_df)
                    
                    logger.info(f"Betting recommendations for race {race_id}:")
                    for _, row in stakes_df.iterrows():
                        if row["stake"] > 0:
                            logger.info(
                                f"Horse {row['horse_id']} ({row['horse_name']}): "
                                f"Stake {row['stake']} yen, "
                                f"Odds {row['odds_win']}, "
                                f"Prob {row['p_blend']:.2f}, "
                                f"ROI {row['ROI']:.2f}, "
                                f"CLV {row['CLV']:.2f}"
                            )
                    
                    timestamp = dt.datetime.now().strftime("%Y%m%d%H%M%S")
                    for _, row in stakes_df.iterrows():
                        if row["stake"] > 0:
                            success = submit_bet(
                                row["horse_id"],
                                race_id,
                                row["stake"],
                                row["odds_win"],
                            )
                            
                            if success:
                                save_bet_record(
                                    race_id,
                                    row["horse_id"],
                                    row["stake"],
                                    row["odds_win"],
                                    timestamp,
                                )
            
            logger.info(f"Sleeping for {RATE_LIMIT_SLEEP} seconds")
            time.sleep(RATE_LIMIT_SLEEP)
            
    except Exception as e:
        logger.exception(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Live betting system")
    parser.add_argument("race_id", nargs="?", default="202504260611", 
                        help="Race ID in format YYYYMMDDCCRRD (default: 202504260611)")
    
    args = parser.parse_args()
    main(args.race_id)
