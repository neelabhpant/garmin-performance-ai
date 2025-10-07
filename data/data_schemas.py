"""Data schemas for Garmin Performance AI using Pydantic."""

from datetime import datetime
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict


class UserProfile(BaseModel):
    """Schema for user profile data."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    user_id: str = Field(..., pattern="^USER_[0-9]{4}$", description="User identifier")
    age: int = Field(..., ge=18, le=70, description="User age in years")
    gender: Literal["M", "F"] = Field(..., description="Gender")
    height_cm: float = Field(..., ge=150, le=200, description="Height in centimeters")
    weight_kg: float = Field(..., ge=45, le=120, description="Weight in kilograms")
    vo2_max_baseline: float = Field(..., ge=30, le=70, description="Baseline VO2 max (ml/kg/min)")
    weekly_miles_baseline: float = Field(..., ge=5, le=80, description="Baseline weekly mileage")
    injury_prone: bool = Field(default=False, description="History of injuries")
    years_running: float = Field(..., ge=0, le=50, description="Years of running experience")
    persona_type: Literal["elite", "competitive", "recreational", "beginner"] = Field(
        ..., description="Runner persona category"
    )


class TrainingData(BaseModel):
    """Schema for daily training data."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    user_id: str = Field(..., pattern="^USER_[0-9]{4}$")
    date: datetime = Field(..., description="Training date")
    workout_type: Literal["easy", "tempo", "intervals", "long", "recovery", "rest"] = Field(
        ..., description="Type of workout"
    )
    distance_km: float = Field(..., ge=0, le=50, description="Distance in kilometers")
    duration_min: float = Field(..., ge=0, le=360, description="Duration in minutes")
    avg_hr: int = Field(..., ge=40, le=220, description="Average heart rate")
    max_hr: int = Field(..., ge=40, le=220, description="Maximum heart rate")
    avg_pace_min_per_km: float = Field(..., ge=0, le=10, description="Average pace (min/km)")
    elevation_gain_m: float = Field(..., ge=0, le=2000, description="Elevation gain in meters")
    hrv_morning: float = Field(..., ge=10, le=200, description="Morning HRV")
    sleep_hours: float = Field(..., ge=3, le=12, description="Hours of sleep")
    stress_score: float = Field(..., ge=0, le=10, description="Subjective stress score")
    temperature_c: float = Field(..., ge=-20, le=45, description="Temperature in Celsius")
    humidity_pct: float = Field(..., ge=0, le=100, description="Humidity percentage")
    fatigue_level: float = Field(..., ge=0, le=10, description="Subjective fatigue level")
    current_vo2_max: float = Field(..., ge=30, le=70, description="Current estimated VO2 max")


class PredictionOutput(BaseModel):
    """Schema for model prediction outputs."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    user_id: str = Field(..., pattern="^USER_[0-9]{4}$")
    prediction_date: datetime = Field(..., description="Date of prediction")
    
    race_5k_seconds: float = Field(..., description="5K finish time in seconds")
    race_10k_seconds: float = Field(..., description="10K finish time in seconds")
    race_half_marathon_seconds: float = Field(..., description="Half marathon finish time")
    race_marathon_seconds: float = Field(..., description="Marathon finish time")
    
    race_5k_formatted: str = Field(..., description="5K time formatted as MM:SS")
    race_10k_formatted: str = Field(..., description="10K time formatted as MM:SS")
    race_half_formatted: str = Field(..., description="Half marathon time as H:MM:SS")
    race_marathon_formatted: str = Field(..., description="Marathon time as H:MM:SS")
    
    readiness_score: int = Field(..., ge=0, le=100, description="Training readiness (0-100)")
    readiness_status: Literal["green", "yellow", "red"] = Field(
        ..., description="Readiness traffic light"
    )
    
    injury_risk_pct: float = Field(..., ge=0, le=100, description="Injury risk percentage")
    injury_risk_level: Literal["Low", "Moderate", "High"] = Field(
        ..., description="Injury risk category"
    )
    injury_concerns: List[str] = Field(default_factory=list, description="Specific injury concerns")
    
    pace_splits_5k: List[float] = Field(..., min_length=5, max_length=5)
    pace_splits_10k: List[float] = Field(..., min_length=10, max_length=10)
    pace_splits_half: List[float] = Field(..., min_length=10, max_length=10)
    pace_splits_marathon: List[float] = Field(..., min_length=10, max_length=10)
    
    vo2_max_current: float = Field(..., ge=30, le=70, description="Current VO2 max")
    vo2_max_30day_projection: float = Field(..., ge=30, le=70, description="30-day VO2 max projection")
    vo2_max_trend: float = Field(..., description="Monthly VO2 max change rate")
    
    explanation: str = Field(..., description="LLM-generated explanation")
    confidence_5k: Tuple[float, float] = Field(..., description="5K time confidence interval")
    confidence_10k: Tuple[float, float] = Field(..., description="10K time confidence interval")
    confidence_half: Tuple[float, float] = Field(..., description="Half marathon confidence interval")
    confidence_marathon: Tuple[float, float] = Field(..., description="Marathon confidence interval")


class FeatureVector(BaseModel):
    """Engineered features for model input."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    user_id: str
    
    acute_load: float = Field(..., description="7-day training load")
    chronic_load: float = Field(..., description="28-day average training load")
    acwr: float = Field(..., description="Acute:Chronic Workload Ratio")
    
    recent_vo2_max: float = Field(..., description="Most recent VO2 max estimate")
    vo2_max_trend: float = Field(..., description="VO2 max change rate")
    
    weekly_mileage: float = Field(..., description="Current weekly mileage")
    mileage_trend: float = Field(..., description="Mileage change rate")
    
    avg_pace_recent: float = Field(..., description="7-day average pace")
    pace_improvement: float = Field(..., description="Pace improvement rate")
    
    hrv_baseline: float = Field(..., description="HRV baseline")
    hrv_cv: float = Field(..., description="HRV coefficient of variation")
    
    sleep_avg: float = Field(..., description="Average sleep hours")
    sleep_consistency: float = Field(..., description="Sleep pattern consistency")
    
    fatigue_accumulated: float = Field(..., description="Accumulated fatigue score")
    recovery_score: float = Field(..., description="Recovery quality score")
    
    tempo_percentage: float = Field(..., description="Percentage of tempo workouts")
    interval_percentage: float = Field(..., description="Percentage of interval workouts")
    long_run_percentage: float = Field(..., description="Percentage of long runs")
    
    age: int
    gender: Literal["M", "F"]
    bmi: float = Field(..., description="Body Mass Index")
    years_running: float
    injury_history: bool