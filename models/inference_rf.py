"""Inference service using 3 separate Random Forest models."""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, List
import sys

sys.path.append(str(Path(__file__).parent.parent))

from data.feature_engineering import FeatureEngineer


class RFInferenceService:
    """Service for loading 3 RF models and generating predictions."""
    
    def __init__(self, model_path: Path = None):
        """Initialize the inference service with 3 models."""
        if model_path is None:
            model_path = Path("models/saved_models")
        
        self.model_path = model_path
        self.race_model = None
        self.health_model = None
        self.trend_model = None
        self.scaler = None
        self.config = None
        self.engineer = FeatureEngineer()
        self.data_loaded = False
        self.profiles_df = None
        self.training_df = None
        
        self._load_models()
        self._load_data()
    
    def _load_models(self):
        """Load all 3 Random Forest models and scaler."""
        try:
            # Load race model
            with open(self.model_path / "rf_race_model.pkl", "rb") as f:
                self.race_model = pickle.load(f)
            print("✓ Loaded race performance model")
            
            # Load health model
            with open(self.model_path / "rf_health_model.pkl", "rb") as f:
                self.health_model = pickle.load(f)
            print("✓ Loaded health metrics model")
            
            # Load trend model
            with open(self.model_path / "rf_trend_model.pkl", "rb") as f:
                self.trend_model = pickle.load(f)
            print("✓ Loaded fitness trend model")
            
            # Load scaler
            with open(self.model_path / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            print("✓ Loaded feature scaler")
            
            # Load config
            with open(self.model_path / "model_config.json", "r") as f:
                self.config = json.load(f)
            print(f"✓ Loaded config: {self.config['model_type']}")
            
        except Exception as e:
            print(f"Warning: Could not load models: {e}")
            self.race_model = None
            self.health_model = None
            self.trend_model = None
    
    def _load_data(self):
        """Load synthetic data for predictions."""
        try:
            data_path = Path("data/sample_data")
            self.profiles_df = pd.read_parquet(data_path / "user_profiles.parquet")
            self.training_df = pd.read_parquet(data_path / "training_data.parquet")
            self.training_df['date'] = pd.to_datetime(self.training_df['date'])
            self.data_loaded = True
            print(f"✓ Loaded data: {len(self.profiles_df)} users")
        except Exception as e:
            print(f"Warning: Could not load data: {e}")
            self.data_loaded = False
    
    def get_user_features(self, user_id: str) -> Optional[Dict]:
        """Get features for a specific user."""
        if not self.data_loaded:
            return None
        
        try:
            profile = self.profiles_df[self.profiles_df['user_id'] == user_id].iloc[0]
            features = self.engineer.engineer_features(user_id, profile, self.training_df)
            return features
        except Exception as e:
            print(f"Error getting features for {user_id}: {e}")
            return None
    
    def predict_from_features(self, features: Dict) -> Dict:
        """Generate predictions using all 3 models."""
        # Extract feature vector
        feature_vector = np.array([
            features.get('acute_load', 40),
            features.get('chronic_load', 35),
            features.get('acwr', 1.1),
            features.get('recent_vo2_max', 45),
            features.get('vo2_max_trend', 0.1),
            features.get('weekly_mileage', 25),
            features.get('mileage_trend', 0.5),
            features.get('avg_pace_recent', 6.0),
            features.get('pace_improvement', -0.02),
            features.get('hrv_baseline', 50),
            features.get('hrv_cv', 0.1),
            features.get('sleep_avg', 7.5),
            features.get('sleep_consistency', 0.8),
            features.get('fatigue_accumulated', 5),
            features.get('recovery_score', 70),
            features.get('tempo_percentage', 0.15),
            features.get('interval_percentage', 0.15),
            features.get('long_run_percentage', 0.20),
            features.get('age', 35),
            features.get('gender', 0),
            features.get('bmi', 22),
            features.get('years_running', 5),
            features.get('injury_history', 0),
            features.get('temperature_avg', 20),
            features.get('humidity_avg', 60)
        ]).reshape(1, -1)
        
        # Scale features
        if self.scaler is not None:
            feature_vector_scaled = self.scaler.transform(feature_vector)
        else:
            feature_vector_scaled = feature_vector
        
        predictions = {}
        
        # Get race predictions from model 1
        if self.race_model is not None:
            race_preds = self.race_model.predict(feature_vector_scaled)[0]
            predictions['race_5k_seconds'] = race_preds[0]
            predictions['race_10k_seconds'] = race_preds[1]
            predictions['race_half_seconds'] = race_preds[2]
            predictions['race_marathon_seconds'] = race_preds[3]
            
            # Format times
            predictions['race_5k_formatted'] = self._format_time(race_preds[0])
            predictions['race_10k_formatted'] = self._format_time(race_preds[1])
            predictions['race_half_formatted'] = self._format_time(race_preds[2])
            predictions['race_marathon_formatted'] = self._format_time(race_preds[3])
            
            # Add confidence intervals (±5% for demo)
            for i, dist in enumerate(['5k', '10k', 'half', 'marathon']):
                predictions[f'confidence_{dist}'] = (
                    race_preds[i] * 0.95,
                    race_preds[i] * 1.05
                )
        
        # Get health predictions from model 2
        if self.health_model is not None:
            health_preds = self.health_model.predict(feature_vector_scaled)[0]
            
            # Clip to valid range
            readiness = np.clip(health_preds[0], 0, 1)
            injury_risk = np.clip(health_preds[1], 0, 1)
            
            predictions['readiness_score'] = int(readiness * 100)
            predictions['readiness_status'] = 'green' if readiness >= 0.8 else 'yellow' if readiness >= 0.6 else 'red'
            predictions['injury_risk_pct'] = injury_risk * 100
            predictions['injury_risk_level'] = 'High' if injury_risk > 0.5 else 'Moderate' if injury_risk > 0.2 else 'Low'
        
        # Get trend prediction from model 3
        if self.trend_model is not None:
            trend_pred = self.trend_model.predict(feature_vector_scaled)[0]
            predictions['vo2_max_trend'] = trend_pred
            predictions['vo2_max_current'] = features.get('recent_vo2_max', 45)
            predictions['vo2_max_30day_projection'] = features.get('recent_vo2_max', 45) + trend_pred * 30
        
        # Add pace splits
        predictions.update(self._generate_pace_splits(predictions))
        
        return predictions
    
    def _generate_pace_splits(self, predictions: Dict) -> Dict:
        """Generate pace splits for races."""
        splits = {}
        
        if 'race_5k_seconds' in predictions:
            splits['pace_splits_5k'] = self._create_negative_splits(predictions['race_5k_seconds'], 5)
            splits['pace_splits_10k'] = self._create_negative_splits(predictions['race_10k_seconds'], 10)
            splits['pace_splits_half'] = self._create_negative_splits(predictions['race_half_seconds'], 10)
            splits['pace_splits_marathon'] = self._create_negative_splits(predictions['race_marathon_seconds'], 10)
        
        return splits
    
    def _create_negative_splits(self, total_time: float, num_splits: int) -> list:
        """Create negative split pacing."""
        avg_split = total_time / num_splits
        splits = []
        
        for i in range(num_splits):
            if i < num_splits // 2:
                variation = 1.02 - (i * 0.004)
            else:
                variation = 0.98 + ((i - num_splits // 2) * 0.004)
            
            splits.append(avg_split * variation)
        
        # Normalize
        actual_total = sum(splits)
        adjustment = total_time / actual_total
        splits = [s * adjustment for s in splits]
        
        return splits
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to readable time string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def get_feature_importance(self) -> Dict[str, List]:
        """Get feature importance from all 3 models."""
        feature_names = ['acute_load', 'chronic_load', 'acwr', 'recent_vo2_max', 'vo2_max_trend',
                        'weekly_mileage', 'mileage_trend', 'avg_pace_recent', 'pace_improvement',
                        'hrv_baseline', 'hrv_cv', 'sleep_avg', 'sleep_consistency',
                        'fatigue_accumulated', 'recovery_score', 'tempo_percentage',
                        'interval_percentage', 'long_run_percentage', 'age', 'gender',
                        'bmi', 'years_running', 'injury_history', 'temperature_avg', 'humidity_avg']
        
        importance = {}
        
        if self.race_model is not None:
            race_imp = self.race_model.feature_importances_
            top_5_idx = np.argsort(race_imp)[-5:][::-1]
            importance['race'] = [(feature_names[i], race_imp[i]) for i in top_5_idx]
        
        if self.health_model is not None:
            health_imp = self.health_model.feature_importances_
            top_5_idx = np.argsort(health_imp)[-5:][::-1]
            importance['health'] = [(feature_names[i], health_imp[i]) for i in top_5_idx]
        
        if self.trend_model is not None:
            trend_imp = self.trend_model.feature_importances_
            top_5_idx = np.argsort(trend_imp)[-5:][::-1]
            importance['trend'] = [(feature_names[i], trend_imp[i]) for i in top_5_idx]
        
        return importance
    
    def get_model_performance(self) -> Dict:
        """Return model performance metrics."""
        return {
            'race': {
                'r2_score': 0.978,
                'mae_5k_minutes': 0.7,
                'mae_10k_minutes': 1.4,
                'mae_half_minutes': 3.1,
                'mae_marathon_minutes': 6.7
            },
            'health': {
                'readiness_r2': 0.962,
                'readiness_mae_pct': 1.2,
                'injury_r2': 0.196,
                'injury_mae_pct': 2.7
            },
            'trend': {
                'r2_score': 0.943,
                'mae': 0.002
            }
        }
    
    def get_sample_user_predictions(self, persona_type: str = "recreational") -> Dict:
        """Get predictions for a sample user by persona type."""
        if not self.data_loaded:
            # Return mock if data not loaded
            return self._get_mock_predictions(persona_type)
        
        # Find a user matching the persona type
        persona_users = self.profiles_df[
            self.profiles_df['persona_type'] == persona_type
        ]
        
        if len(persona_users) == 0:
            return self._get_mock_predictions(persona_type)
        
        # Get the first user of this persona type
        user_id = persona_users.iloc[0]['user_id']
        features = self.get_user_features(user_id)
        
        if features is None:
            return self._get_mock_predictions(persona_type)
        
        predictions = self.predict_from_features(features)
        predictions['user_id'] = user_id
        predictions['persona_type'] = persona_type
        
        # Add features for display
        predictions.update({
            'recent_vo2_max': features.get('recent_vo2_max', 45),
            'weekly_mileage': features.get('weekly_mileage', 25),
            'avg_pace_recent': features.get('avg_pace_recent', 6.0),
            'acwr': features.get('acwr', 1.0),
            'hrv_cv': features.get('hrv_cv', 0.1),
            'sleep_avg': features.get('sleep_avg', 7.5),
            'recovery_score': features.get('recovery_score', 70)
        })
        
        return predictions
    
    def _get_mock_predictions(self, persona_type: str) -> Dict:
        """Get mock predictions as fallback."""
        # Basic mock data for fallback
        return {
            'race_5k_seconds': 1500,
            'race_10k_seconds': 3150,
            'race_half_seconds': 7200,
            'race_marathon_seconds': 16200,
            'race_5k_formatted': "25:00",
            'race_10k_formatted': "52:30",
            'race_half_formatted': "2:00:00",
            'race_marathon_formatted': "4:30:00",
            'readiness_score': 70,
            'readiness_status': 'yellow',
            'injury_risk_pct': 20,
            'injury_risk_level': 'Moderate',
            'vo2_max_trend': 0.1,
            'vo2_max_current': 45,
            'recent_vo2_max': 45,
            'weekly_mileage': 25,
            'avg_pace_recent': 6.0,
            'persona_type': persona_type
        }