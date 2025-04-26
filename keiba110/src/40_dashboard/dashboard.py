"""
Streamlit dashboard for keiba110.
Shows all KPIs including betting performance, model metrics, and race predictions.
"""

import datetime as dt
import json
import logging
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = pathlib.Path("data/raw")
PROCESSED_DATA_DIR = pathlib.Path("data/processed")
MODEL_DIR = pathlib.Path("data/model")
BET_DIR = pathlib.Path("data/bets")


def load_bets() -> pd.DataFrame:
    """
    Load bet records from CSV file.
    
    Returns:
        DataFrame with bet records
    """
    bet_file = BET_DIR / "bets.csv"
    
    if not bet_file.exists():
        logger.warning(f"Bet file not found: {bet_file}")
        return pd.DataFrame(columns=["timestamp", "race_id", "horse_id", "stake", "odds"])
    
    try:
        df = pd.read_csv(bet_file)
        logger.info(f"Loaded {len(df)} bet records")
        return df
    except Exception as e:
        logger.error(f"Error loading bet records: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "race_id", "horse_id", "stake", "odds"])


def load_race_results() -> pd.DataFrame:
    """
    Load race results from CSV file.
    
    Returns:
        DataFrame with race results
    """
    results_file = PROCESSED_DATA_DIR / "race_results.csv"
    
    if not results_file.exists():
        logger.warning(f"Results file not found: {results_file}")
        return pd.DataFrame(columns=["race_id", "horse_id", "position", "payouts"])
    
    try:
        df = pd.read_csv(results_file)
        logger.info(f"Loaded results for {df['race_id'].nunique()} races")
        return df
    except Exception as e:
        logger.error(f"Error loading race results: {str(e)}")
        return pd.DataFrame(columns=["race_id", "horse_id", "position", "payouts"])


def load_features() -> pd.DataFrame:
    """
    Load features from Parquet file.
    
    Returns:
        DataFrame with features
    """
    features_file = PROCESSED_DATA_DIR / "features.parquet"
    
    if not features_file.exists():
        logger.warning(f"Features file not found: {features_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(features_file)
        logger.info(f"Loaded features with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        return pd.DataFrame()


def calculate_performance_metrics(bets: pd.DataFrame, results: pd.DataFrame) -> Dict:
    """
    Calculate performance metrics from bet records and race results.
    
    Args:
        bets: DataFrame with bet records
        results: DataFrame with race results
    
    Returns:
        Dictionary with performance metrics
    """
    if bets.empty or results.empty:
        return {
            "total_bets": 0,
            "total_stake": 0,
            "total_return": 0,
            "profit": 0,
            "roi": 0,
            "win_rate": 0,
            "avg_odds": 0,
            "avg_stake": 0,
            "max_drawdown": 0,
        }
    
    merged = pd.merge(
        bets,
        results[["race_id", "horse_id", "position", "payouts"]],
        on=["race_id", "horse_id"],
        how="left",
    )
    
    merged["won"] = merged["position"] == 1
    merged["return"] = merged.apply(
        lambda x: x["stake"] * x["odds"] if x["won"] else 0, axis=1
    )
    merged["profit"] = merged["return"] - merged["stake"]
    
    merged = merged.sort_values("timestamp")
    merged["cum_stake"] = merged["stake"].cumsum()
    merged["cum_return"] = merged["return"].cumsum()
    merged["cum_profit"] = merged["profit"].cumsum()
    merged["cum_roi"] = merged["cum_return"] / merged["cum_stake"] - 1
    
    merged["peak"] = merged["cum_profit"].cummax()
    merged["drawdown"] = merged["peak"] - merged["cum_profit"]
    
    total_bets = len(merged)
    total_stake = merged["stake"].sum()
    total_return = merged["return"].sum()
    profit = total_return - total_stake
    roi = total_return / total_stake - 1 if total_stake > 0 else 0
    win_rate = merged["won"].mean()
    avg_odds = merged["odds"].mean()
    avg_stake = merged["stake"].mean()
    max_drawdown = merged["drawdown"].max()
    
    return {
        "total_bets": total_bets,
        "total_stake": total_stake,
        "total_return": total_return,
        "profit": profit,
        "roi": roi,
        "win_rate": win_rate,
        "avg_odds": avg_odds,
        "avg_stake": avg_stake,
        "max_drawdown": max_drawdown,
        "performance_df": merged,
    }


def display_header() -> None:
    """
    Display dashboard header.
    """
    st.set_page_config(
        page_title="Keiba110 Dashboard",
        page_icon="🏇",
        layout="wide",
    )
    
    st.title("🏇 Keiba110 Dashboard")
    st.markdown("Horse racing prediction and betting system")
    
    st.sidebar.markdown(f"**Last updated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


def display_performance_metrics(metrics: Dict) -> None:
    """
    Display performance metrics.
    
    Args:
        metrics: Dictionary with performance metrics
    """
    st.header("📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Profit",
            value=f"¥{metrics['profit']:,.0f}",
            delta=f"{metrics['roi'] * 100:.1f}% ROI",
        )
    
    with col2:
        st.metric(
            label="Win Rate",
            value=f"{metrics['win_rate'] * 100:.1f}%",
            delta=f"{metrics['total_bets']} bets",
        )
    
    with col3:
        st.metric(
            label="Average Odds",
            value=f"{metrics['avg_odds']:.2f}",
        )
    
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"¥{metrics['max_drawdown']:,.0f}",
        )
    
    if "performance_df" in metrics and not metrics["performance_df"].empty:
        df = metrics["performance_df"]
        
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["cum_profit"],
                mode="lines",
                name="Cumulative Profit",
                line=dict(color="green", width=2),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["profit"],
                mode="markers",
                name="Individual Bets",
                marker=dict(
                    color=df["profit"].apply(lambda x: "green" if x > 0 else "red"),
                    size=8,
                ),
            )
        )
        
        fig.update_layout(
            title="Profit Over Time",
            xaxis_title="Date",
            yaxis_title="Profit (¥)",
            height=400,
        )
        
        st.plotly_chart(fig, use_container_width=True)


def display_recent_bets(bets: pd.DataFrame, results: pd.DataFrame, features: pd.DataFrame) -> None:
    """
    Display recent bets.
    
    Args:
        bets: DataFrame with bet records
        results: DataFrame with race results
        features: DataFrame with features
    """
    st.header("🎯 Recent Bets")
    
    if bets.empty:
        st.info("No bet records found")
        return
    
    merged = pd.merge(
        bets,
        results[["race_id", "horse_id", "position", "payouts"]],
        on=["race_id", "horse_id"],
        how="left",
    )
    
    if not features.empty:
        horse_names = features[["race_id", "horse_id", "horse_name"]].drop_duplicates()
        merged = pd.merge(
            merged,
            horse_names,
            on=["race_id", "horse_id"],
            how="left",
        )
    
    merged["won"] = merged["position"] == 1
    merged["return"] = merged.apply(
        lambda x: x["stake"] * x["odds"] if x["won"] else 0, axis=1
    )
    merged["profit"] = merged["return"] - merged["stake"]
    
    merged = merged.sort_values("timestamp", ascending=False)
    
    recent_bets = merged.head(10)
    
    display_df = recent_bets[["timestamp", "race_id", "horse_name", "stake", "odds", "position", "profit"]].copy()
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    display_df["profit"] = display_df["profit"].apply(lambda x: f"¥{x:,.0f}")
    display_df["stake"] = display_df["stake"].apply(lambda x: f"¥{x:,.0f}")
    display_df["position"] = display_df["position"].fillna("Pending")
    
    display_df.columns = ["Timestamp", "Race ID", "Horse", "Stake", "Odds", "Position", "Profit"]
    
    st.dataframe(display_df, use_container_width=True)


def display_model_metrics() -> None:
    """
    Display model performance metrics.
    """
    st.header("🧠 Model Performance")
    
    metrics_file = MODEL_DIR / "model_metrics.json"
    
    if not metrics_file.exists():
        st.info("No model metrics found")
        return
    
    try:
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="LightGBM AUC",
                value=f"{metrics.get('lgb_auc', 0):.4f}",
            )
        
        with col2:
            st.metric(
                label="XGBoost AUC",
                value=f"{metrics.get('xgb_auc', 0):.4f}",
            )
        
        with col3:
            st.metric(
                label="Blender AUC",
                value=f"{metrics.get('blend_auc', 0):.4f}",
            )
        
        if "feature_importance" in metrics:
            importance_df = pd.DataFrame(
                metrics["feature_importance"],
                columns=["Feature", "Importance"],
            )
            importance_df = importance_df.sort_values("Importance", ascending=False).head(15)
            
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top 15 Feature Importances",
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        logger.error(f"Error loading model metrics: {str(e)}")
        st.error("Error loading model metrics")


def display_upcoming_races(features: pd.DataFrame) -> None:
    """
    Display upcoming races.
    
    Args:
        features: DataFrame with features
    """
    st.header("🏁 Upcoming Races")
    
    if features.empty:
        st.info("No race data found")
        return
    
    races = features[["race_id", "date", "course"]].drop_duplicates()
    
    races["date"] = pd.to_datetime(races["date"], format="%Y%m%d")
    
    today = dt.datetime.now().date()
    upcoming = races[races["date"].dt.date >= today].sort_values("date")
    
    if upcoming.empty:
        st.info("No upcoming races found")
        return
    
    course_names = {
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
    
    upcoming["course_name"] = upcoming["course"].map(course_names)
    
    display_df = upcoming[["race_id", "date", "course_name"]].copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    
    display_df.columns = ["Race ID", "Date", "Course"]
    
    st.dataframe(display_df, use_container_width=True)
    
    selected_race = st.selectbox(
        "Select a race to view details",
        options=upcoming["race_id"].tolist(),
        format_func=lambda x: f"{x} ({course_names.get(x[8:10], '')})",
    )
    
    if selected_race:
        display_race_details(selected_race, features)


def display_race_details(race_id: str, features: pd.DataFrame) -> None:
    """
    Display details for a specific race.
    
    Args:
        race_id: Race ID
        features: DataFrame with features
    """
    st.subheader(f"Race Details: {race_id}")
    
    race_df = features[features["race_id"] == race_id].copy()
    
    if race_df.empty:
        st.info(f"No data found for race {race_id}")
        return
    
    has_predictions = "p_blend" in race_df.columns
    
    if has_predictions:
        race_df = race_df.sort_values("p_blend", ascending=False)
    elif "odds_win" in race_df.columns:
        race_df = race_df.sort_values("odds_win")
    else:
        race_df = race_df.sort_values("draw")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "turf_state" in race_df.columns:
            turf_state = race_df["turf_state"].iloc[0]
            st.metric("Turf Condition", turf_state)
        
        if "dirt_state" in race_df.columns:
            dirt_state = race_df["dirt_state"].iloc[0]
            st.metric("Dirt Condition", dirt_state)
    
    with col2:
        if "wx" in race_df.columns:
            weather = race_df["wx"].iloc[0]
            st.metric("Weather", weather)
        
        if "temp_max" in race_df.columns and "temp_min" in race_df.columns:
            temp_max = race_df["temp_max"].iloc[0]
            temp_min = race_df["temp_min"].iloc[0]
            if pd.notna(temp_max) and pd.notna(temp_min):
                st.metric("Temperature", f"{temp_min}°C - {temp_max}°C")
    
    st.subheader("Horses")
    
    display_cols = ["horse_name", "sex_age", "jockey", "weight", "draw"]
    
    if "odds_win" in race_df.columns:
        display_cols.append("odds_win")
    
    if has_predictions:
        display_cols.extend(["p_blend", "ROI", "CLV", "stake"])
    
    display_df = race_df[display_cols].copy()
    
    column_names = {
        "horse_name": "Horse",
        "sex_age": "Sex/Age",
        "jockey": "Jockey",
        "weight": "Weight",
        "draw": "Draw",
        "odds_win": "Odds",
        "p_blend": "Win Prob",
        "ROI": "ROI",
        "CLV": "CLV",
        "stake": "Stake",
    }
    
    display_df.rename(columns=column_names, inplace=True)
    
    if "Win Prob" in display_df.columns:
        display_df["Win Prob"] = display_df["Win Prob"].apply(lambda x: f"{x:.2f}")
    
    if "ROI" in display_df.columns:
        display_df["ROI"] = display_df["ROI"].apply(lambda x: f"{x:.2f}")
    
    if "CLV" in display_df.columns:
        display_df["CLV"] = display_df["CLV"].apply(lambda x: f"{x:.2f}")
    
    if "Stake" in display_df.columns:
        display_df["Stake"] = display_df["Stake"].apply(lambda x: f"¥{x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True)
    
    if has_predictions:
        st.subheader("Betting Recommendations")
        
        bet_df = race_df[race_df["stake"] > 0].copy()
        
        if bet_df.empty:
            st.info("No betting recommendations for this race")
        else:
            for _, row in bet_df.iterrows():
                st.markdown(
                    f"**{row['horse_name']}** - Stake: ¥{row['stake']:,.0f}, "
                    f"Odds: {row['odds_win']}, Prob: {row['p_blend']:.2f}, "
                    f"ROI: {row['ROI']:.2f}, CLV: {row['CLV']:.2f}"
                )


def main() -> None:
    """
    Main function to run the dashboard.
    """
    display_header()
    
    bets = load_bets()
    results = load_race_results()
    features = load_features()
    
    metrics = calculate_performance_metrics(bets, results)
    
    display_performance_metrics(metrics)
    display_recent_bets(bets, results, features)
    display_model_metrics()
    display_upcoming_races(features)


if __name__ == "__main__":
    main()
