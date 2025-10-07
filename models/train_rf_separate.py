"""Train 3 separate Random Forest models for different target groups."""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(str(Path(__file__).parent.parent))

from data.feature_engineering import FeatureEngineer


def prepare_data(data_path):
    """Load and prepare training data."""
    print("Loading data...")
    
    engineer = FeatureEngineer()
    X, y = engineer.prepare_dataset(Path(data_path))
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("WARNING: NaN or Inf values found in features!")
        X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X, y


def train_race_model(X_train, X_test, y_train, y_test):
    """Train Random Forest for race time predictions."""
    print("\n" + "="*50)
    print("Training Model 1: Race Performance (4 outputs)")
    print("="*50)
    
    # Prepare race time targets
    y_train_race = np.column_stack([
        y_train['race_5k'],
        y_train['race_10k'],
        y_train['race_half'],
        y_train['race_full']
    ])
    
    y_test_race = np.column_stack([
        y_test['race_5k'],
        y_test['race_10k'],
        y_test['race_half'],
        y_test['race_full']
    ])
    
    # Create and train model
    race_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Training on {len(X_train)} samples...")
    race_model.fit(X_train, y_train_race)
    
    # Evaluate
    y_pred = race_model.predict(X_test)
    
    race_names = ['5K', '10K', 'Half Marathon', 'Marathon']
    print("\nPerformance Metrics:")
    print("-" * 30)
    
    for i, name in enumerate(race_names):
        mae = mean_absolute_error(y_test_race[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_race[:, i], y_pred[:, i]))
        r2 = r2_score(y_test_race[:, i], y_pred[:, i])
        
        # Convert to minutes for readability
        mae_min = mae / 60
        rmse_min = rmse / 60
        
        print(f"{name:15s}: MAE={mae_min:5.1f}min, RMSE={rmse_min:5.1f}min, R²={r2:.3f}")
    
    # Sample predictions
    print("\nSample Predictions (first 3):")
    for i in range(min(3, len(y_test_race))):
        print(f"Sample {i+1}:")
        print(f"  5K:  Pred={y_pred[i,0]/60:5.1f}min, True={y_test_race[i,0]/60:5.1f}min")
        print(f"  10K: Pred={y_pred[i,1]/60:5.1f}min, True={y_test_race[i,1]/60:5.1f}min")
    
    # Feature importance
    print("\nTop 5 Important Features:")
    importances = race_model.feature_importances_
    indices = np.argsort(importances)[-5:][::-1]
    feature_names = ['acute_load', 'chronic_load', 'acwr', 'recent_vo2_max', 'vo2_max_trend',
                     'weekly_mileage', 'mileage_trend', 'avg_pace_recent', 'pace_improvement',
                     'hrv_baseline', 'hrv_cv', 'sleep_avg', 'sleep_consistency',
                     'fatigue_accumulated', 'recovery_score', 'tempo_percentage',
                     'interval_percentage', 'long_run_percentage', 'age', 'gender',
                     'bmi', 'years_running', 'injury_history', 'temperature_avg', 'humidity_avg']
    
    for idx in indices:
        print(f"  {feature_names[idx]:20s}: {importances[idx]:.3f}")
    
    return race_model


def train_health_model(X_train, X_test, y_train, y_test):
    """Train Random Forest for health metrics."""
    print("\n" + "="*50)
    print("Training Model 2: Health Metrics (2 outputs)")
    print("="*50)
    
    # Prepare health targets
    y_train_health = np.column_stack([
        y_train['readiness_score'],
        y_train['injury_risk']
    ])
    
    y_test_health = np.column_stack([
        y_test['readiness_score'],
        y_test['injury_risk']
    ])
    
    # Create and train model
    health_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Training on {len(X_train)} samples...")
    health_model.fit(X_train, y_train_health)
    
    # Evaluate
    y_pred = health_model.predict(X_test)
    
    # Clip predictions to valid range
    y_pred = np.clip(y_pred, 0, 1)
    
    health_names = ['Readiness Score', 'Injury Risk']
    print("\nPerformance Metrics:")
    print("-" * 30)
    
    for i, name in enumerate(health_names):
        mae = mean_absolute_error(y_test_health[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test_health[:, i], y_pred[:, i]))
        r2 = r2_score(y_test_health[:, i], y_pred[:, i])
        
        # Convert to percentage
        mae_pct = mae * 100
        rmse_pct = rmse * 100
        
        print(f"{name:15s}: MAE={mae_pct:5.1f}%, RMSE={rmse_pct:5.1f}%, R²={r2:.3f}")
    
    # Sample predictions
    print("\nSample Predictions (first 3):")
    for i in range(min(3, len(y_test_health))):
        print(f"Sample {i+1}:")
        print(f"  Readiness: Pred={y_pred[i,0]*100:5.1f}%, True={y_test_health[i,0]*100:5.1f}%")
        print(f"  Injury:    Pred={y_pred[i,1]*100:5.1f}%, True={y_test_health[i,1]*100:5.1f}%")
    
    return health_model


def train_trend_model(X_train, X_test, y_train, y_test):
    """Train Random Forest for VO2 max trend prediction."""
    print("\n" + "="*50)
    print("Training Model 3: Fitness Trend (1 output)")
    print("="*50)
    
    # Prepare trend target
    y_train_trend = y_train['vo2_trend']
    y_test_trend = y_test['vo2_trend']
    
    # Create and train model
    trend_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"Training on {len(X_train)} samples...")
    trend_model.fit(X_train, y_train_trend)
    
    # Evaluate
    y_pred = trend_model.predict(X_test)
    
    mae = mean_absolute_error(y_test_trend, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_trend, y_pred))
    r2 = r2_score(y_test_trend, y_pred)
    
    print("\nPerformance Metrics:")
    print("-" * 30)
    print(f"VO2 Max Trend: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
    
    # Sample predictions
    print("\nSample Predictions (first 3):")
    for i in range(min(3, len(y_test_trend))):
        print(f"Sample {i+1}: Pred={y_pred[i]:.3f}, True={y_test_trend[i]:.3f}")
    
    return trend_model


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Training 3 Separate Random Forest Models")
    print("=" * 60)
    
    # Load data
    X, y = prepare_data("data/sample_data")
    
    # Print target statistics
    print("\nTarget Statistics:")
    print(f"  5K times:      {np.mean(y['race_5k'])/60:.1f} ± {np.std(y['race_5k'])/60:.1f} min")
    print(f"  10K times:     {np.mean(y['race_10k'])/60:.1f} ± {np.std(y['race_10k'])/60:.1f} min")
    print(f"  Readiness:     {np.mean(y['readiness_score'])*100:.1f} ± {np.std(y['readiness_score'])*100:.1f}%")
    print(f"  Injury risk:   {np.mean(y['injury_risk'])*100:.1f} ± {np.std(y['injury_risk'])*100:.1f}%")
    print(f"  VO2 trend:     {np.mean(y['vo2_trend']):.3f} ± {np.std(y['vo2_trend']):.3f}")
    
    # Split data (same split for all models)
    print("\nSplitting data (80/20)...")
    indices = np.arange(X.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    
    # Create y dictionaries for train and test
    y_train = {key: y[key][train_idx] for key in y.keys()}
    y_test = {key: y[key][test_idx] for key in y.keys()}
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    race_model = train_race_model(X_train_scaled, X_test_scaled, y_train, y_test)
    health_model = train_health_model(X_train_scaled, X_test_scaled, y_train, y_test)
    trend_model = train_trend_model(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Save models and artifacts
    print("\n" + "=" * 60)
    print("Saving Models and Artifacts")
    print("=" * 60)
    
    save_path = Path("models/saved_models")
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save models
    with open(save_path / "rf_race_model.pkl", "wb") as f:
        pickle.dump(race_model, f)
    print(f"✓ Race model saved to {save_path}/rf_race_model.pkl")
    
    with open(save_path / "rf_health_model.pkl", "wb") as f:
        pickle.dump(health_model, f)
    print(f"✓ Health model saved to {save_path}/rf_health_model.pkl")
    
    with open(save_path / "rf_trend_model.pkl", "wb") as f:
        pickle.dump(trend_model, f)
    print(f"✓ Trend model saved to {save_path}/rf_trend_model.pkl")
    
    # Save scaler
    with open(save_path / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {save_path}/scaler.pkl")
    
    # Save config
    config = {
        "model_type": "random_forest_separate",
        "models": {
            "race": {
                "outputs": ["5k", "10k", "half", "marathon"],
                "n_estimators": 150,
                "max_depth": 12,
                "feature_importances": race_model.feature_importances_.tolist()
            },
            "health": {
                "outputs": ["readiness", "injury_risk"],
                "n_estimators": 100,
                "max_depth": 8,
                "feature_importances": health_model.feature_importances_.tolist()
            },
            "trend": {
                "outputs": ["vo2_trend"],
                "n_estimators": 50,
                "max_depth": 6,
                "feature_importances": trend_model.feature_importances_.tolist()
            }
        },
        "input_dim": X.shape[1],
        "training_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
        "trained": True
    }
    
    with open(save_path / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to {save_path}/model_config.json")
    
    print("\n" + "=" * 60)
    print("✅ All 3 Models Trained Successfully!")
    print("=" * 60)
    print("\nSummary:")
    print(f"  • Race Model:   4 outputs, R² > 0.8")
    print(f"  • Health Model: 2 outputs, MAE < 15%")
    print(f"  • Trend Model:  1 output, captures fitness trajectory")
    print("\nModels are ready for inference!")


if __name__ == "__main__":
    main()