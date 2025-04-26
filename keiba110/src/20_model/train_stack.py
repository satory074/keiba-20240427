"""
Train stacked ensemble model.
LightGBM + XGBoost + Logit → Isotonic calibration → stacking blender.
"""

import logging
import pathlib
import pickle
from typing import Dict, List, Optional, Tuple, Union

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = pathlib.Path("data/processed")
MODEL_DIR = pathlib.Path("data/model")
RANDOM_STATE = 42
N_FOLDS = 5
TEST_SIZE = 0.2


def load_features() -> pd.DataFrame:
    """
    Load features from Parquet file.
    
    Returns:
        DataFrame with features
    """
    features_file = PROCESSED_DATA_DIR / "features.parquet"
    
    if not features_file.exists():
        logger.error(f"Features file not found: {features_file}")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(features_file)
        logger.info(f"Loaded features with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading features: {str(e)}")
        return pd.DataFrame()


def prepare_data(df: pd.DataFrame, target_col: str = "is_favorite") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling.
    
    Args:
        df: DataFrame with features
        target_col: Target column name
    
    Returns:
        Tuple of (X, y) for modeling
    """
    if df.empty:
        return pd.DataFrame(), pd.Series()
    
    result = df.copy()
    
    if target_col not in result.columns:
        logger.error(f"Target column not found: {target_col}")
        return pd.DataFrame(), pd.Series()
    
    y = result[target_col]
    
    drop_cols = [
        "race_id", "horse_id", "horse_name", "jockey", "trainer", "owner",
        target_col,
    ]
    X = result.drop([col for col in drop_cols if col in result.columns], axis=1)
    
    X = X.fillna(X.mean())
    
    logger.info(f"Prepared data with shape: {X.shape}")
    
    return X, y


def train_lightgbm(X: pd.DataFrame, y: pd.Series) -> Tuple[lgb.Booster, List[float]]:
    """
    Train LightGBM model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
    
    Returns:
        Tuple of (model, feature_importances)
    """
    logger.info("Training LightGBM model")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": RANDOM_STATE,
    }
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    
    feature_importances = model.feature_importance(importance_type="gain")
    
    y_pred = model.predict(X_val)
    auc = roc_auc_score(y_val, y_pred)
    loss = log_loss(y_val, y_pred)
    
    logger.info(f"LightGBM - AUC: {auc:.4f}, Log Loss: {loss:.4f}")
    
    return model, feature_importances


def train_xgboost(X: pd.DataFrame, y: pd.Series) -> Tuple[xgb.Booster, Dict[str, float]]:
    """
    Train XGBoost model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
    
    Returns:
        Tuple of (model, feature_importances)
    """
    logger.info("Training XGBoost model")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "seed": RANDOM_STATE,
    }
    
    model = xgb.train(
        params,
        train_data,
        num_boost_round=1000,
        evals=[(train_data, "train"), (val_data, "val")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    
    feature_importances = model.get_score(importance_type="gain")
    
    y_pred = model.predict(val_data)
    auc = roc_auc_score(y_val, y_pred)
    loss = log_loss(y_val, y_pred)
    
    logger.info(f"XGBoost - AUC: {auc:.4f}, Log Loss: {loss:.4f}")
    
    return model, feature_importances


def train_logistic_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[LogisticRegression, np.ndarray]:
    """
    Train Logistic Regression model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
    
    Returns:
        Tuple of (model, feature_importances)
    """
    logger.info("Training Logistic Regression model")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, y_train)
    
    feature_importances = np.abs(model.coef_[0])
    
    y_pred = model.predict_proba(X_val_scaled)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    loss = log_loss(y_val, y_pred)
    
    logger.info(f"Logistic Regression - AUC: {auc:.4f}, Log Loss: {loss:.4f}")
    
    model.scaler = scaler
    
    return model, feature_importances


def train_isotonic_calibration(X: pd.DataFrame, y: pd.Series, base_model: Union[lgb.Booster, xgb.Booster, LogisticRegression],
                              model_type: str) -> IsotonicRegression:
    """
    Train Isotonic Calibration model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        base_model: Base model to calibrate
        model_type: Type of base model ("lgb", "xgb", or "logit")
    
    Returns:
        Calibration model
    """
    logger.info(f"Training Isotonic Calibration for {model_type}")
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    if model_type == "lgb":
        y_pred = base_model.predict(X_val)
    elif model_type == "xgb":
        val_data = xgb.DMatrix(X_val)
        y_pred = base_model.predict(val_data)
    elif model_type == "logit":
        X_val_scaled = base_model.scaler.transform(X_val)
        y_pred = base_model.predict_proba(X_val_scaled)[:, 1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred, y_val)
    
    y_cal = calibrator.predict(y_pred)
    auc = roc_auc_score(y_val, y_cal)
    loss = log_loss(y_val, y_cal)
    
    logger.info(f"Calibrated {model_type} - AUC: {auc:.4f}, Log Loss: {loss:.4f}")
    
    return calibrator


def train_stacking_blender(X: pd.DataFrame, y: pd.Series, 
                          lgb_model: lgb.Booster, xgb_model: xgb.Booster, logit_model: LogisticRegression,
                          lgb_calibrator: IsotonicRegression, xgb_calibrator: IsotonicRegression, logit_calibrator: IsotonicRegression) -> LogisticRegression:
    """
    Train stacking blender model.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        lgb_model: LightGBM model
        xgb_model: XGBoost model
        logit_model: Logistic Regression model
        lgb_calibrator: LightGBM calibrator
        xgb_calibrator: XGBoost calibrator
        logit_calibrator: Logistic Regression calibrator
    
    Returns:
        Stacking blender model
    """
    logger.info("Training stacking blender")
    
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    lgb_preds = np.zeros(len(X))
    xgb_preds = np.zeros(len(X))
    logit_preds = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_fold = lgb.train(
            lgb_model.params,
            lgb_train,
            num_boost_round=lgb_model.best_iteration,
        )
        lgb_fold_preds = lgb_fold.predict(X_val)
        lgb_preds[val_idx] = lgb_calibrator.predict(lgb_fold_preds)
        
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_val = xgb.DMatrix(X_val)
        xgb_fold = xgb.train(
            xgb_model.get_xgb_params(),
            xgb_train,
            num_boost_round=xgb_model.best_iteration,
        )
        xgb_fold_preds = xgb_fold.predict(xgb_val)
        xgb_preds[val_idx] = xgb_calibrator.predict(xgb_fold_preds)
        
        X_train_scaled = logit_model.scaler.transform(X_train)
        X_val_scaled = logit_model.scaler.transform(X_val)
        logit_fold = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
        logit_fold.fit(X_train_scaled, y_train)
        logit_fold_preds = logit_fold.predict_proba(X_val_scaled)[:, 1]
        logit_preds[val_idx] = logit_calibrator.predict(logit_fold_preds)
    
    meta_features = np.column_stack([lgb_preds, xgb_preds, logit_preds])
    
    blender = LogisticRegression(C=1.0, random_state=RANDOM_STATE)
    blender.fit(meta_features, y)
    
    y_pred = blender.predict_proba(meta_features)[:, 1]
    auc = roc_auc_score(y, y_pred)
    loss = log_loss(y, y_pred)
    
    logger.info(f"Stacking Blender - AUC: {auc:.4f}, Log Loss: {loss:.4f}")
    
    return blender


def save_models(lgb_model: lgb.Booster, xgb_model: xgb.Booster, logit_model: LogisticRegression,
               lgb_calibrator: IsotonicRegression, xgb_calibrator: IsotonicRegression, logit_calibrator: IsotonicRegression,
               blender: LogisticRegression, feature_names: List[str]) -> None:
    """
    Save trained models to disk.
    
    Args:
        lgb_model: LightGBM model
        xgb_model: XGBoost model
        logit_model: Logistic Regression model
        lgb_calibrator: LightGBM calibrator
        xgb_calibrator: XGBoost calibrator
        logit_calibrator: Logistic Regression calibrator
        blender: Stacking blender model
        feature_names: List of feature names
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    lgb_model.save_model(str(MODEL_DIR / "lgb_model.txt"))
    
    xgb_model.save_model(str(MODEL_DIR / "xgb_model.json"))
    
    with open(MODEL_DIR / "logit_model.pkl", "wb") as f:
        pickle.dump(logit_model, f)
    
    with open(MODEL_DIR / "lgb_calibrator.pkl", "wb") as f:
        pickle.dump(lgb_calibrator, f)
    
    with open(MODEL_DIR / "xgb_calibrator.pkl", "wb") as f:
        pickle.dump(xgb_calibrator, f)
    
    with open(MODEL_DIR / "logit_calibrator.pkl", "wb") as f:
        pickle.dump(logit_calibrator, f)
    
    with open(MODEL_DIR / "blender.pkl", "wb") as f:
        pickle.dump(blender, f)
    
    with open(MODEL_DIR / "feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    
    logger.info(f"Saved all models to {MODEL_DIR}")


def main() -> None:
    """
    Main function to train stacked ensemble model.
    """
    df = load_features()
    
    if df.empty:
        logger.error("No features available for training")
        return
    
    X, y = prepare_data(df)
    
    if X.empty or len(y) == 0:
        logger.error("Failed to prepare data for training")
        return
    
    lgb_model, lgb_importances = train_lightgbm(X, y)
    xgb_model, xgb_importances = train_xgboost(X, y)
    logit_model, logit_importances = train_logistic_regression(X, y)
    
    lgb_calibrator = train_isotonic_calibration(X, y, lgb_model, "lgb")
    xgb_calibrator = train_isotonic_calibration(X, y, xgb_model, "xgb")
    logit_calibrator = train_isotonic_calibration(X, y, logit_model, "logit")
    
    blender = train_stacking_blender(
        X, y,
        lgb_model, xgb_model, logit_model,
        lgb_calibrator, xgb_calibrator, logit_calibrator,
    )
    
    save_models(
        lgb_model, xgb_model, logit_model,
        lgb_calibrator, xgb_calibrator, logit_calibrator,
        blender, X.columns.tolist(),
    )
    
    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()
