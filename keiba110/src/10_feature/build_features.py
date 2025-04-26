"""
Build features from raw data.
Creates a 40-column Parquet feature set with mutual information filtering.
"""

import logging
import pathlib
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = pathlib.Path("data/raw")
PROCESSED_DATA_DIR = pathlib.Path("data/processed")
MI_THRESHOLD = 0.01  # Mutual information threshold for feature selection


def load_entries_data() -> pd.DataFrame:
    """
    Load race entries data from Parquet files.
    
    Returns:
        DataFrame with race entries data
    """
    entries_files = list(RAW_DATA_DIR.glob("*_entries.parquet"))
    
    if not entries_files:
        logger.warning("No entries files found")
        return pd.DataFrame()
    
    dfs = []
    for file in entries_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading entries file {file}: {str(e)}")
    
    if not dfs:
        logger.warning("No entries data loaded")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def load_odds_data() -> pd.DataFrame:
    """
    Load odds data from JSON files.
    
    Returns:
        DataFrame with odds data
    """
    odds_files = list(RAW_DATA_DIR.glob("*_odds_*.json"))
    
    if not odds_files:
        logger.warning("No odds files found")
        return pd.DataFrame()
    
    data = []
    for file in odds_files:
        try:
            filename = file.name
            race_id = filename.split("_")[0]
            
            odds_data = pd.read_json(file, orient="records")
            
            odds_data["race_id"] = race_id
            
            timestamp = filename.split("_")[-1].split(".")[0]
            odds_data["timestamp"] = timestamp
            
            data.append(odds_data)
        except Exception as e:
            logger.error(f"Error loading odds file {file}: {str(e)}")
    
    if not data:
        logger.warning("No odds data loaded")
        return pd.DataFrame()
    
    return pd.concat(data, ignore_index=True)


def load_baba_data() -> pd.DataFrame:
    """
    Load track condition data from Parquet files.
    
    Returns:
        DataFrame with track condition data
    """
    baba_files = list(RAW_DATA_DIR.glob("*_baba.parquet"))
    
    if not baba_files:
        logger.warning("No track condition files found")
        return pd.DataFrame()
    
    dfs = []
    for file in baba_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading track condition file {file}: {str(e)}")
    
    if not dfs:
        logger.warning("No track condition data loaded")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def load_weather_data() -> pd.DataFrame:
    """
    Load weather data from Parquet files.
    
    Returns:
        DataFrame with weather data
    """
    weather_files = list(RAW_DATA_DIR.glob("weather_*.parquet"))
    
    if not weather_files:
        logger.warning("No weather files found")
        return pd.DataFrame()
    
    dfs = []
    for file in weather_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading weather file {file}: {str(e)}")
    
    if not dfs:
        logger.warning("No weather data loaded")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def load_gansui_data() -> pd.DataFrame:
    """
    Load annual moisture data from Parquet files.
    
    Returns:
        DataFrame with annual moisture data
    """
    gansui_files = list(RAW_DATA_DIR.glob("gansui_*.parquet"))
    
    if not gansui_files:
        logger.warning("No annual moisture files found")
        return pd.DataFrame()
    
    dfs = []
    for file in gansui_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Error loading annual moisture file {file}: {str(e)}")
    
    if not dfs:
        logger.warning("No annual moisture data loaded")
        return pd.DataFrame()
    
    return pd.concat(dfs, ignore_index=True)


def preprocess_entries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess race entries data.
    
    Args:
        df: DataFrame with race entries data
    
    Returns:
        Preprocessed DataFrame
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if "sex_age" in result.columns:
        result["sex"] = result["sex_age"].str[0]
        result["age"] = result["sex_age"].str[1:].astype(int)
    
    if "draw" in result.columns:
        result["draw"] = pd.to_numeric(result["draw"], errors="coerce").astype("Int64")
    
    if "weight" in result.columns:
        result["weight"] = pd.to_numeric(result["weight"], errors="coerce").astype("Int64")
    
    return result


def preprocess_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess odds data.
    
    Args:
        df: DataFrame with odds data
    
    Returns:
        Preprocessed DataFrame
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    for col in result.columns:
        if "odds" in col.lower():
            result[col] = pd.to_numeric(result[col], errors="coerce")
    
    return result


def preprocess_baba(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess track condition data.
    
    Args:
        df: DataFrame with track condition data
    
    Returns:
        Preprocessed DataFrame
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if "turf_state" in result.columns:
        result["turf_state"] = result["turf_state"].astype("category")
    
    if "dirt_state" in result.columns:
        result["dirt_state"] = result["dirt_state"].astype("category")
    
    return result


def preprocess_weather(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess weather data.
    
    Args:
        df: DataFrame with weather data
    
    Returns:
        Preprocessed DataFrame
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if "wx" in result.columns:
        result["wx"] = result["wx"].astype("category")
    
    return result


def preprocess_gansui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess annual moisture data.
    
    Args:
        df: DataFrame with annual moisture data
    
    Returns:
        Preprocessed DataFrame
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    return result


def merge_data(entries: pd.DataFrame, odds: pd.DataFrame, baba: pd.DataFrame, 
               weather: pd.DataFrame, gansui: pd.DataFrame) -> pd.DataFrame:
    """
    Merge all data sources into a single DataFrame.
    
    Args:
        entries: DataFrame with race entries data
        odds: DataFrame with odds data
        baba: DataFrame with track condition data
        weather: DataFrame with weather data
        gansui: DataFrame with annual moisture data
    
    Returns:
        Merged DataFrame
    """
    if entries.empty:
        logger.warning("No entries data available for merging")
        return pd.DataFrame()
    
    result = entries.copy()
    
    if not odds.empty:
        latest_odds = odds.sort_values("timestamp", ascending=False).drop_duplicates(
            subset=["race_id", "horse_id"], keep="first"
        )
        
        result = pd.merge(
            result,
            latest_odds[["race_id", "horse_id", "odds_win", "odds_plc_low", "odds_plc_high"]],
            on=["race_id", "horse_id"],
            how="left",
        )
    
    result["date"] = result["race_id"].str[:8]
    result["course"] = result["race_id"].str[8:10]
    
    if not baba.empty:
        result = pd.merge(
            result,
            baba[["date", "course", "turf_state", "dirt_state", "moisture_front", "cushion"]],
            on=["date", "course"],
            how="left",
        )
    
    if not weather.empty:
        latest_weather = weather.sort_values("ts", ascending=False).drop_duplicates(
            subset=["area"], keep="first"
        )
        
        course_to_area = {
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
        
        result["area"] = result["course"].map(course_to_area)
        
        result = pd.merge(
            result,
            latest_weather[["area", "wx", "temp_max", "temp_min", "wind"]],
            on="area",
            how="left",
        )
        
        result.drop("area", axis=1, inplace=True)
    
    if not gansui.empty:
        result = pd.merge(
            result,
            gansui[["date", "course", "turf_moist", "dirt_moist"]],
            on=["date", "course"],
            how="left",
        )
    
    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer additional features.
    
    Args:
        df: DataFrame with merged data
    
    Returns:
        DataFrame with engineered features
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    if "odds_win" in result.columns and "odds_plc_low" in result.columns:
        result["odds_ratio"] = result["odds_win"] / result["odds_plc_low"]
    
    if "draw" in result.columns:
        draw_stats = result.groupby("race_id")["draw"].agg(["min", "max", "count"]).reset_index()
        draw_stats.columns = ["race_id", "min_draw", "max_draw", "num_horses"]
        
        result = pd.merge(result, draw_stats, on="race_id", how="left")
        
        result["draw_norm"] = (result["draw"] - result["min_draw"]) / (result["max_draw"] - result["min_draw"])
        
        result["draw_inside"] = (result["draw"] <= result["num_horses"] / 2).astype(int)
        
        result.drop(["min_draw", "max_draw"], axis=1, inplace=True)
    
    cat_columns = ["sex", "turf_state", "dirt_state", "wx"]
    for col in cat_columns:
        if col in result.columns:
            dummies = pd.get_dummies(result[col], prefix=col, drop_first=True)
            
            result = pd.concat([result, dummies], axis=1)
    
    if "weight" in result.columns:
        result["weight_pct"] = result.groupby("race_id")["weight"].transform(
            lambda x: (x.rank(pct=True) * 100).round() / 100
        )
    
    if "odds_win" in result.columns:
        result["odds_rank"] = result.groupby("race_id")["odds_win"].transform(
            lambda x: x.rank(method="min")
        )
        
        result["odds_rank_norm"] = result.groupby("race_id")["odds_rank"].transform(
            lambda x: (x - 1) / (x.max() - 1)
        )
        
        result["is_favorite"] = (result["odds_rank"] == 1).astype(int)
    
    return result


def select_features(df: pd.DataFrame, target_col: str = "odds_win") -> pd.DataFrame:
    """
    Select features based on mutual information with target.
    
    Args:
        df: DataFrame with all features
        target_col: Target column for mutual information calculation
    
    Returns:
        DataFrame with selected features
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    numeric_cols = result.select_dtypes(include=["number"]).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    X = result[numeric_cols].fillna(0)
    y = result[target_col].fillna(0)
    
    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({"feature": numeric_cols, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False)
    
    logger.info("Mutual information scores:")
    for _, row in mi_df.iterrows():
        logger.info(f"{row['feature']}: {row['mi_score']:.4f}")
    
    selected_features = mi_df[mi_df["mi_score"] >= MI_THRESHOLD]["feature"].tolist()
    
    keep_cols = ["race_id", "horse_id", "horse_name", "jockey", "trainer", "owner"]
    selected_features.extend([col for col in keep_cols if col in result.columns])
    
    if target_col not in selected_features and target_col in result.columns:
        selected_features.append(target_col)
    
    if len(selected_features) < 40 and len(result.columns) >= 40:
        additional_features = mi_df[~mi_df["feature"].isin(selected_features)]["feature"].tolist()
        selected_features.extend(additional_features[:40 - len(selected_features)])
    
    result = result[selected_features]
    
    logger.info(f"Selected {len(selected_features)} features")
    
    return result


def main() -> None:
    """
    Main function to build features.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading data from all sources")
    entries_df = load_entries_data()
    odds_df = load_odds_data()
    baba_df = load_baba_data()
    weather_df = load_weather_data()
    gansui_df = load_gansui_data()
    
    logger.info("Preprocessing data")
    entries_df = preprocess_entries(entries_df)
    odds_df = preprocess_odds(odds_df)
    baba_df = preprocess_baba(baba_df)
    weather_df = preprocess_weather(weather_df)
    gansui_df = preprocess_gansui(gansui_df)
    
    logger.info("Merging data")
    merged_df = merge_data(entries_df, odds_df, baba_df, weather_df, gansui_df)
    
    if merged_df.empty:
        logger.error("No data available after merging")
        return
    
    logger.info("Engineering features")
    features_df = engineer_features(merged_df)
    
    logger.info("Selecting features")
    selected_df = select_features(features_df)
    
    output_file = PROCESSED_DATA_DIR / "features.parquet"
    selected_df.to_parquet(output_file, index=False)
    
    logger.info(f"Saved features to {output_file}")
    logger.info(f"Feature set shape: {selected_df.shape}")


if __name__ == "__main__":
    main()
