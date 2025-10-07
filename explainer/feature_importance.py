"""Feature importance analysis and human-readable explanations."""

from typing import Dict, List, Tuple, Optional
import numpy as np


class FeatureExplainer:
    """Convert model features to human-readable explanations."""
    
    # Feature name to human-readable mapping
    FEATURE_DESCRIPTIONS = {
        'acute_load': 'Recent Training Load (7 days)',
        'chronic_load': 'Long-term Training Load (28 days)',
        'acwr': 'Acute:Chronic Workload Ratio',
        'recent_vo2_max': 'Current VO2 Max',
        'vo2_max_trend': 'VO2 Max Trend',
        'weekly_mileage': 'Weekly Running Distance',
        'mileage_trend': 'Mileage Change Rate',
        'avg_pace_recent': 'Average Recent Pace',
        'pace_improvement': 'Pace Improvement Rate',
        'hrv_baseline': 'Heart Rate Variability Baseline',
        'hrv_cv': 'HRV Consistency',
        'sleep_avg': 'Average Sleep Hours',
        'sleep_consistency': 'Sleep Pattern Consistency',
        'fatigue_accumulated': 'Accumulated Fatigue',
        'recovery_score': 'Recovery Quality Score',
        'tempo_percentage': 'Tempo Training Percentage',
        'interval_percentage': 'Interval Training Percentage',
        'long_run_percentage': 'Long Run Percentage',
        'age': 'Age',
        'gender': 'Gender',
        'bmi': 'Body Mass Index',
        'years_running': 'Running Experience (years)',
        'injury_history': 'Previous Injury Count',
        'temperature_avg': 'Average Training Temperature',
        'humidity_avg': 'Average Training Humidity'
    }
    
    # Feature impact explanations
    FEATURE_IMPACTS = {
        'acute_load': {
            'high': 'Your recent training load is elevated, which can boost fitness but requires careful recovery.',
            'low': 'Your recent training load is low, allowing for increased intensity if feeling fresh.',
            'optimal': 'Your recent training load is well-balanced for adaptation.'
        },
        'acwr': {
            'high': 'Your training load has spiked recently - high injury risk. Consider reducing intensity.',
            'low': 'Your training has been very light recently - you may be detraining.',
            'optimal': 'Your training load progression is ideal for safe improvement.'
        },
        'recent_vo2_max': {
            'high': 'Your aerobic fitness is excellent, supporting fast race times.',
            'low': 'Your aerobic fitness has room for improvement through consistent training.',
            'optimal': 'Your VO2 max is good for your age and experience level.'
        },
        'recovery_score': {
            'high': 'You\'re recovering well from training - ready for quality work.',
            'low': 'Your recovery is compromised - focus on rest and easy efforts.',
            'optimal': 'Your recovery is adequate but could be improved.'
        }
    }
    
    @classmethod
    def get_feature_description(cls, feature_name: str) -> str:
        """Get human-readable description of a feature."""
        return cls.FEATURE_DESCRIPTIONS.get(feature_name, feature_name.replace('_', ' ').title())
    
    @classmethod
    def explain_feature_importance(cls, feature_importances: List[Tuple[str, float]], 
                                  top_n: int = 5) -> str:
        """Generate explanation of top feature importances.
        
        Args:
            feature_importances: List of (feature_name, importance) tuples
            top_n: Number of top features to explain
            
        Returns:
            Human-readable explanation
        """
        if not feature_importances:
            return "No feature importance data available."
        
        # Sort by importance
        sorted_features = sorted(feature_importances, key=lambda x: x[1], reverse=True)[:top_n]
        
        explanations = ["**Key Performance Factors:**\n"]
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            description = cls.get_feature_description(feature)
            pct = importance * 100
            explanations.append(f"{i}. **{description}** ({pct:.1f}% impact)")
        
        return '\n'.join(explanations)
    
    @classmethod
    def explain_feature_value(cls, feature_name: str, value: float, 
                            percentile: Optional[float] = None) -> str:
        """Explain what a specific feature value means.
        
        Args:
            feature_name: Name of the feature
            value: Current value
            percentile: Percentile rank (0-100) if available
            
        Returns:
            Explanation of what this value means
        """
        description = cls.get_feature_description(feature_name)
        
        # Special handling for key features
        if feature_name == 'acwr':
            if value < 0.8:
                status = 'low'
                explanation = 'Risk of detraining'
            elif value > 1.5:
                status = 'high'
                explanation = 'High injury risk'
            else:
                status = 'optimal'
                explanation = 'Optimal training load'
                
            return f"{description}: {value:.2f} - {explanation}"
        
        elif feature_name == 'recent_vo2_max':
            if value > 55:
                level = 'Excellent'
            elif value > 45:
                level = 'Good'
            elif value > 35:
                level = 'Fair'
            else:
                level = 'Below Average'
                
            return f"{description}: {value:.1f} ml/kg/min - {level}"
        
        elif feature_name == 'recovery_score':
            if value > 80:
                status = 'Excellent recovery'
            elif value > 60:
                status = 'Good recovery'
            else:
                status = 'Needs more recovery'
                
            return f"{description}: {value:.0f}% - {status}"
        
        # Default formatting
        if percentile:
            return f"{description}: {value:.2f} (Top {100-percentile:.0f}%)"
        else:
            return f"{description}: {value:.2f}"
    
    @classmethod
    def get_training_focus_from_features(cls, features: Dict) -> List[str]:
        """Determine training focus areas from feature values.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            List of recommended focus areas
        """
        focus_areas = []
        
        # Check VO2 max
        if features.get('recent_vo2_max', 0) < 40:
            focus_areas.append("Build aerobic base with easy runs")
        elif features.get('recent_vo2_max', 0) > 50:
            focus_areas.append("Maintain aerobic fitness, add speed work")
        
        # Check ACWR
        acwr = features.get('acwr', 1.0)
        if acwr < 0.8:
            focus_areas.append("Gradually increase training volume")
        elif acwr > 1.3:
            focus_areas.append("Reduce volume to prevent injury")
        
        # Check recovery
        if features.get('recovery_score', 0) < 60:
            focus_areas.append("Prioritize recovery and sleep")
        
        # Check mileage trend
        if features.get('mileage_trend', 0) > 10:
            focus_areas.append("Stabilize mileage increase")
        
        # Check sleep
        if features.get('sleep_avg', 0) < 7:
            focus_areas.append("Improve sleep duration and quality")
        
        return focus_areas if focus_areas else ["Maintain current balanced approach"]
    
    @classmethod
    def generate_feature_summary(cls, features: Dict) -> str:
        """Generate a summary of key features in natural language.
        
        Args:
            features: Dictionary of feature values
            
        Returns:
            Natural language summary
        """
        summaries = []
        
        # Training load summary
        acwr = features.get('acwr', 1.0)
        if acwr < 0.8:
            summaries.append("Training load is very light")
        elif acwr > 1.5:
            summaries.append("Training load is dangerously high")
        elif 1.0 <= acwr <= 1.3:
            summaries.append("Training load is progressing well")
        
        # Fitness summary
        vo2 = features.get('recent_vo2_max', 45)
        if vo2 > 50:
            summaries.append(f"excellent aerobic fitness (VO2: {vo2:.1f})")
        elif vo2 > 40:
            summaries.append(f"good aerobic fitness (VO2: {vo2:.1f})")
        else:
            summaries.append(f"developing aerobic fitness (VO2: {vo2:.1f})")
        
        # Recovery summary
        recovery = features.get('recovery_score', 70)
        if recovery > 80:
            summaries.append("recovering very well")
        elif recovery < 60:
            summaries.append("showing signs of poor recovery")
        
        # Combine into sentence
        if summaries:
            return f"You have {', '.join(summaries)}."
        else:
            return "Your training metrics are within normal ranges."