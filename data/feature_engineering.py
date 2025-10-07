"""Feature engineering pipeline for model training."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from data.data_schemas import FeatureVector


class FeatureEngineer:
    """Extract and engineer features from raw training data."""
    
    def __init__(self):
        """Initialize feature engineering pipeline."""
        self.features_cache = {}
    
    def load_data(self, data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load user profiles and training data."""
        profiles_df = pd.read_parquet(data_path / "user_profiles.parquet")
        training_df = pd.read_parquet(data_path / "training_data.parquet")
        training_df['date'] = pd.to_datetime(training_df['date'])
        return profiles_df, training_df
    
    def calculate_training_loads(self, user_data: pd.DataFrame) -> Dict:
        """Calculate acute and chronic training loads."""
        user_data = user_data.sort_values('date')
        
        user_data['training_load'] = (
            user_data['distance_km'] * 
            (user_data['avg_pace_min_per_km'] / 6.0) * 
            (user_data['avg_hr'] / 150)
        )
        
        last_7_days = user_data.tail(7)
        last_28_days = user_data.tail(28)
        
        acute_load = last_7_days['training_load'].sum()
        chronic_load = last_28_days['training_load'].sum() / 4
        acwr = acute_load / chronic_load if chronic_load > 0 else 1.0
        
        return {
            'acute_load': acute_load,
            'chronic_load': chronic_load,
            'acwr': min(2.0, max(0.5, acwr))
        }
    
    def calculate_vo2_metrics(self, user_data: pd.DataFrame) -> Dict:
        """Calculate VO2 max related metrics."""
        user_data = user_data.sort_values('date')
        
        recent_vo2 = user_data['current_vo2_max'].iloc[-1]
        
        if len(user_data) >= 30:
            month_ago_vo2 = user_data['current_vo2_max'].iloc[-30]
            vo2_trend = (recent_vo2 - month_ago_vo2) / 30
        else:
            vo2_trend = 0.1
        
        return {
            'recent_vo2_max': recent_vo2,
            'vo2_max_trend': vo2_trend
        }
    
    def calculate_pace_metrics(self, user_data: pd.DataFrame) -> Dict:
        """Calculate pace-related metrics."""
        user_data = user_data.sort_values('date')
        
        last_7_days = user_data[user_data['distance_km'] > 0].tail(7)
        
        if len(last_7_days) > 0:
            avg_pace_recent = last_7_days['avg_pace_min_per_km'].mean()
            
            if len(user_data) >= 30:
                month_ago_pace = user_data[user_data['distance_km'] > 0].iloc[-30:-23]['avg_pace_min_per_km'].mean()
                pace_improvement = (month_ago_pace - avg_pace_recent) / month_ago_pace
            else:
                pace_improvement = 0.01
        else:
            avg_pace_recent = 6.0
            pace_improvement = 0
        
        return {
            'avg_pace_recent': avg_pace_recent,
            'pace_improvement': pace_improvement
        }
    
    def calculate_recovery_metrics(self, user_data: pd.DataFrame) -> Dict:
        """Calculate recovery and fatigue metrics."""
        user_data = user_data.sort_values('date')
        
        hrv_baseline = user_data['hrv_morning'].median()
        hrv_std = user_data['hrv_morning'].std()
        hrv_cv = hrv_std / hrv_baseline if hrv_baseline > 0 else 0.1
        
        sleep_avg = user_data['sleep_hours'].mean()
        sleep_std = user_data['sleep_hours'].std()
        sleep_consistency = 1 - (sleep_std / sleep_avg) if sleep_avg > 0 else 0.5
        
        fatigue_accumulated = user_data.tail(7)['fatigue_level'].mean()
        
        recovery_score = 100
        recovery_score -= fatigue_accumulated * 10
        recovery_score -= max(0, (8 - sleep_avg) * 10)
        recovery_score = max(0, min(100, recovery_score))
        
        return {
            'hrv_baseline': hrv_baseline,
            'hrv_cv': hrv_cv,
            'sleep_avg': sleep_avg,
            'sleep_consistency': sleep_consistency,
            'fatigue_accumulated': fatigue_accumulated,
            'recovery_score': recovery_score
        }
    
    def calculate_workout_distribution(self, user_data: pd.DataFrame) -> Dict:
        """Calculate workout type distribution."""
        total_workouts = len(user_data[user_data['workout_type'] != 'rest'])
        
        if total_workouts > 0:
            tempo_pct = len(user_data[user_data['workout_type'] == 'tempo']) / total_workouts
            interval_pct = len(user_data[user_data['workout_type'] == 'intervals']) / total_workouts
            long_pct = len(user_data[user_data['workout_type'] == 'long']) / total_workouts
        else:
            tempo_pct = interval_pct = long_pct = 0
        
        return {
            'tempo_percentage': tempo_pct,
            'interval_percentage': interval_pct,
            'long_run_percentage': long_pct
        }
    
    def calculate_environmental_metrics(self, user_data: pd.DataFrame) -> Dict:
        """Calculate environmental condition averages."""
        return {
            'temperature_avg': user_data['temperature_c'].mean(),
            'humidity_avg': user_data['humidity_pct'].mean()
        }
    
    def calculate_mileage_metrics(self, user_data: pd.DataFrame) -> Dict:
        """Calculate mileage-related metrics."""
        user_data = user_data.sort_values('date')
        
        last_7_days = user_data.tail(7)
        weekly_mileage = last_7_days['distance_km'].sum() * 0.621371
        
        if len(user_data) >= 14:
            prev_week = user_data.iloc[-14:-7]
            prev_weekly_mileage = prev_week['distance_km'].sum() * 0.621371
            mileage_trend = (weekly_mileage - prev_weekly_mileage) / max(prev_weekly_mileage, 1)
        else:
            mileage_trend = 0.1
        
        return {
            'weekly_mileage': weekly_mileage,
            'mileage_trend': mileage_trend
        }
    
    def engineer_features(self, user_id: str, profile: pd.Series, training_data: pd.DataFrame) -> Dict:
        """Engineer all features for a user."""
        user_training = training_data[training_data['user_id'] == user_id]
        
        if len(user_training) < 7:
            return None
        
        features = {}
        
        features.update(self.calculate_training_loads(user_training))
        features.update(self.calculate_vo2_metrics(user_training))
        features.update(self.calculate_pace_metrics(user_training))
        features.update(self.calculate_recovery_metrics(user_training))
        features.update(self.calculate_workout_distribution(user_training))
        features.update(self.calculate_environmental_metrics(user_training))
        features.update(self.calculate_mileage_metrics(user_training))
        
        features['user_id'] = user_id
        features['age'] = profile['age']
        features['gender'] = 1 if profile['gender'] == 'F' else 0
        features['bmi'] = profile['weight_kg'] / ((profile['height_cm'] / 100) ** 2)
        features['years_running'] = profile['years_running']
        features['injury_history'] = 1 if profile['injury_prone'] else 0
        
        return features
    
    def create_training_labels(self, features: Dict, profile: pd.Series) -> Dict:
        """Create training labels based on formulas and profile."""
        vo2_max = features['recent_vo2_max']
        
        velocity_at_vo2max = (vo2_max - 3.5) / 0.182
        
        race_5k = 5000 / (velocity_at_vo2max * 0.95 * 0.06) * 3.6
        race_10k = 10000 / (velocity_at_vo2max * 0.92 * 0.06) * 3.6
        race_half = 21097.5 / (velocity_at_vo2max * 0.88 * 0.06) * 3.6
        race_full = 42195 / (velocity_at_vo2max * 0.83 * 0.06) * 3.6
        
        # More realistic readiness calculation
        base_readiness = 75  # Start at 75% instead of 100%
        
        # Penalties
        if features['acwr'] > 1.3:
            base_readiness -= (features['acwr'] - 1.3) * 30
        elif features['acwr'] < 0.8:
            base_readiness -= (0.8 - features['acwr']) * 20
        
        # Fatigue penalty
        base_readiness -= (features['fatigue_accumulated'] - 5) * 3
        
        # Sleep bonus/penalty
        sleep_diff = features['sleep_avg'] - 7.5
        base_readiness += sleep_diff * 5
        
        # Recovery score influence
        recovery_factor = features.get('recovery_score', 70) / 100
        base_readiness = base_readiness * (0.5 + 0.5 * recovery_factor)
        
        readiness = min(100, max(0, base_readiness)) / 100
        
        injury_risk = 0
        if features['acwr'] > 1.5:
            injury_risk = 0.7
        elif features['acwr'] > 1.3:
            injury_risk = 0.4
        elif features['acwr'] < 0.8:
            injury_risk = 0.3
        else:
            injury_risk = 0.15
        
        if profile['injury_prone']:
            injury_risk *= 1.5
        
        injury_risk = min(1.0, injury_risk)
        
        return {
            'race_5k': race_5k,
            'race_10k': race_10k,
            'race_half': race_half,
            'race_full': race_full,
            'readiness_score': readiness,
            'injury_risk': injury_risk,
            'vo2_trend': features['vo2_max_trend']
        }
    
    def prepare_dataset(self, data_path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare complete dataset for training."""
        profiles_df, training_df = self.load_data(data_path)
        
        X_list = []
        y_dict = {
            'race_5k': [],
            'race_10k': [],
            'race_half': [],
            'race_full': [],
            'readiness_score': [],
            'injury_risk': [],
            'vo2_trend': []
        }
        
        for _, profile in profiles_df.iterrows():
            features = self.engineer_features(profile['user_id'], profile, training_df)
            
            if features is None:
                continue
            
            labels = self.create_training_labels(features, profile)
            
            feature_vector = [
                features['acute_load'],
                features['chronic_load'],
                features['acwr'],
                features['recent_vo2_max'],
                features['vo2_max_trend'],
                features['weekly_mileage'],
                features['mileage_trend'],
                features['avg_pace_recent'],
                features['pace_improvement'],
                features['hrv_baseline'],
                features['hrv_cv'],
                features['sleep_avg'],
                features['sleep_consistency'],
                features['fatigue_accumulated'],
                features['recovery_score'],
                features['tempo_percentage'],
                features['interval_percentage'],
                features['long_run_percentage'],
                features['age'],
                features['gender'],
                features['bmi'],
                features['years_running'],
                features['injury_history'],
                features['temperature_avg'],
                features['humidity_avg']
            ]
            
            X_list.append(feature_vector)
            
            for key in y_dict:
                y_dict[key].append(labels[key])
        
        X = np.array(X_list)
        y = {key: np.array(values) for key, values in y_dict.items()}
        
        return X, y