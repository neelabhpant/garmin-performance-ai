"""Multi-output neural network for performance prediction."""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from typing import Dict, Tuple, List
import json


class PerformancePredictor:
    """Neural network model for multi-output race performance prediction."""
    
    def __init__(self, input_dim: int = 25):
        """Initialize the model architecture."""
        self.input_dim = input_dim
        self.model = None
        self.scaler_params = None
        self.build_model()
        
    def build_model(self):
        """Build multi-output neural network architecture."""
        inputs = keras.Input(shape=(self.input_dim,), name='features')
        
        shared = layers.Dense(128, activation='relu', name='shared_1')(inputs)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        shared = layers.Dense(64, activation='relu', name='shared_2')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.2)(shared)
        
        shared = layers.Dense(32, activation='relu', name='shared_3')(shared)
        
        race_branch = layers.Dense(16, activation='relu', name='race_specific')(shared)
        race_5k = layers.Dense(1, name='race_5k', activation='linear')(race_branch)
        race_10k = layers.Dense(1, name='race_10k', activation='linear')(race_branch)
        race_half = layers.Dense(1, name='race_half', activation='linear')(race_branch)
        race_full = layers.Dense(1, name='race_full', activation='linear')(race_branch)
        
        readiness_branch = layers.Dense(16, activation='relu', name='readiness_branch')(shared)
        readiness = layers.Dense(1, name='readiness_score', activation='sigmoid')(readiness_branch)
        
        injury_branch = layers.Dense(16, activation='relu', name='injury_branch')(shared)
        injury = layers.Dense(1, name='injury_risk', activation='sigmoid')(injury_branch)
        
        vo2_branch = layers.Dense(16, activation='relu', name='vo2_branch')(shared)
        vo2_trend = layers.Dense(1, name='vo2_trend', activation='linear')(vo2_branch)
        
        self.model = models.Model(
            inputs=inputs,
            outputs=[
                race_5k, race_10k, race_half, race_full,
                readiness, injury, vo2_trend
            ]
        )
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'race_5k': 'mse',
                'race_10k': 'mse', 
                'race_half': 'mse',
                'race_full': 'mse',
                'readiness_score': 'binary_crossentropy',
                'injury_risk': 'binary_crossentropy',
                'vo2_trend': 'mse'
            },
            loss_weights={
                'race_5k': 1.0,
                'race_10k': 1.0,
                'race_half': 1.0,
                'race_full': 1.0,
                'readiness_score': 2.0,
                'injury_risk': 3.0,
                'vo2_trend': 1.5
            },
            metrics={
                'race_5k': ['mae'],
                'race_10k': ['mae'],
                'race_half': ['mae'],
                'race_full': ['mae'],
                'readiness_score': ['accuracy'],
                'injury_risk': ['accuracy'],
                'vo2_trend': ['mae']
            }
        )
    
    def prepare_features(self, data: Dict) -> np.ndarray:
        """Convert raw data to feature vector."""
        features = np.array([
            data.get('acute_load', 0),
            data.get('chronic_load', 0),
            data.get('acwr', 1.0),
            data.get('recent_vo2_max', 45),
            data.get('vo2_max_trend', 0),
            data.get('weekly_mileage', 20),
            data.get('mileage_trend', 0),
            data.get('avg_pace_recent', 6.0),
            data.get('pace_improvement', 0),
            data.get('hrv_baseline', 50),
            data.get('hrv_cv', 0.1),
            data.get('sleep_avg', 7.5),
            data.get('sleep_consistency', 0.8),
            data.get('fatigue_accumulated', 5),
            data.get('recovery_score', 50),
            data.get('tempo_percentage', 0.15),
            data.get('interval_percentage', 0.15),
            data.get('long_run_percentage', 0.20),
            data.get('age', 35),
            data.get('gender', 0),
            data.get('bmi', 22),
            data.get('years_running', 5),
            data.get('injury_history', 0),
            data.get('temperature_avg', 20),
            data.get('humidity_avg', 60)
        ])
        
        return features.reshape(1, -1)
    
    def predict(self, features: np.ndarray) -> Dict:
        """Generate predictions from features."""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        predictions = self.model.predict(features, verbose=0)
        
        race_5k_seconds = float(predictions[0][0, 0])
        race_10k_seconds = float(predictions[1][0, 0])
        race_half_seconds = float(predictions[2][0, 0])
        race_marathon_seconds = float(predictions[3][0, 0])
        readiness = float(predictions[4][0, 0]) * 100
        injury_risk = float(predictions[5][0, 0]) * 100
        vo2_trend = float(predictions[6][0, 0])
        
        return {
            'race_5k_seconds': race_5k_seconds,
            'race_10k_seconds': race_10k_seconds,
            'race_half_seconds': race_half_seconds,
            'race_marathon_seconds': race_marathon_seconds,
            'race_5k_formatted': self._format_time(race_5k_seconds),
            'race_10k_formatted': self._format_time(race_10k_seconds),
            'race_half_formatted': self._format_time(race_half_seconds),
            'race_marathon_formatted': self._format_time(race_marathon_seconds),
            'readiness_score': int(readiness),
            'readiness_status': self._get_readiness_status(readiness),
            'injury_risk_pct': injury_risk,
            'injury_risk_level': self._get_injury_level(injury_risk),
            'vo2_max_trend': vo2_trend,
            'pace_splits_5k': self._generate_pace_splits(race_5k_seconds, 5),
            'pace_splits_10k': self._generate_pace_splits(race_10k_seconds, 10),
            'pace_splits_half': self._generate_pace_splits(race_half_seconds, 10),
            'pace_splits_marathon': self._generate_pace_splits(race_marathon_seconds, 10),
            'confidence_5k': (race_5k_seconds * 0.95, race_5k_seconds * 1.05),
            'confidence_10k': (race_10k_seconds * 0.95, race_10k_seconds * 1.05),
            'confidence_half': (race_half_seconds * 0.95, race_half_seconds * 1.05),
            'confidence_marathon': (race_marathon_seconds * 0.95, race_marathon_seconds * 1.05)
        }
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to readable time string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
    def _get_readiness_status(self, score: float) -> str:
        """Convert readiness score to traffic light status."""
        if score >= 80:
            return "green"
        elif score >= 60:
            return "yellow"
        else:
            return "red"
    
    def _get_injury_level(self, risk: float) -> str:
        """Convert injury risk percentage to level."""
        if risk < 20:
            return "Low"
        elif risk < 50:
            return "Moderate"
        else:
            return "High"
    
    def _generate_pace_splits(self, total_seconds: float, num_splits: int) -> List[float]:
        """Generate pace splits for a race with slight negative split."""
        avg_split = total_seconds / num_splits
        
        splits = []
        for i in range(num_splits):
            if i < num_splits // 2:
                variation = 1.02 - (i * 0.005)
            else:
                variation = 0.98 + ((i - num_splits // 2) * 0.005)
            
            splits.append(avg_split * variation)
        
        total_split_time = sum(splits)
        adjustment = total_seconds / total_split_time
        splits = [s * adjustment for s in splits]
        
        return splits
    
    def save_model(self, path: str):
        """Save model weights and architecture."""
        if self.model:
            self.model.save(f"{path}/performance_model.h5")
            
            with open(f"{path}/model_config.json", 'w') as f:
                json.dump({
                    'input_dim': self.input_dim,
                    'scaler_params': self.scaler_params
                }, f)
    
    def load_model(self, path: str):
        """Load model weights and architecture."""
        self.model = keras.models.load_model(f"{path}/performance_model.h5")
        
        with open(f"{path}/model_config.json", 'r') as f:
            config = json.load(f)
            self.input_dim = config['input_dim']
            self.scaler_params = config.get('scaler_params')
    
    def get_formula_baseline(self, vo2_max: float, distance: str) -> float:
        """Calculate baseline predictions using VO2max formulas."""
        velocity_at_vo2max = (vo2_max - 3.5) / 0.182
        
        efficiency_factors = {
            '5k': 0.95,
            '10k': 0.92,
            'half': 0.88,
            'marathon': 0.83
        }
        
        distances_km = {
            '5k': 5,
            '10k': 10,
            'half': 21.0975,
            'marathon': 42.195
        }
        
        eff_factor = efficiency_factors.get(distance, 0.9)
        race_velocity_mpm = velocity_at_vo2max * eff_factor
        
        race_velocity_kmh = race_velocity_mpm * 0.06
        
        race_time_hours = distances_km[distance] / race_velocity_kmh
        race_time_seconds = race_time_hours * 3600
        
        return race_time_seconds