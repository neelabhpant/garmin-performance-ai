"""Generate synthetic training data for Garmin Performance AI POC."""

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from data_schemas import UserProfile, TrainingData


class SyntheticDataGenerator:
    """Generate realistic synthetic training data for runners."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
    def generate_user_profiles(self, num_users: int = 100) -> List[UserProfile]:
        """Generate diverse user profiles across different personas."""
        profiles = []
        
        persona_distribution = {
            "beginner": 0.3,
            "recreational": 0.4,
            "competitive": 0.25,
            "elite": 0.05
        }
        
        for i in range(num_users):
            user_id = f"USER_{i:04d}"
            
            persona = np.random.choice(
                list(persona_distribution.keys()),
                p=list(persona_distribution.values())
            )
            
            if persona == "elite":
                age = random.randint(22, 35)
                vo2_max = random.uniform(60, 70)
                weekly_miles = random.uniform(50, 80)
                years_running = random.uniform(8, 20)
                injury_prone = random.random() < 0.3
            elif persona == "competitive":
                age = random.randint(25, 45)
                vo2_max = random.uniform(50, 60)
                weekly_miles = random.uniform(30, 50)
                years_running = random.uniform(5, 15)
                injury_prone = random.random() < 0.25
            elif persona == "recreational":
                age = random.randint(25, 55)
                vo2_max = random.uniform(40, 50)
                weekly_miles = random.uniform(15, 30)
                years_running = random.uniform(2, 10)
                injury_prone = random.random() < 0.2
            else:
                age = random.randint(20, 60)
                vo2_max = random.uniform(30, 40)
                weekly_miles = random.uniform(5, 15)
                years_running = random.uniform(0, 3)
                injury_prone = random.random() < 0.15
            
            gender = random.choice(["M", "F"])
            
            if gender == "M":
                height = random.gauss(175, 7)
                weight = random.gauss(70, 8)
            else:
                height = random.gauss(165, 6)
                weight = random.gauss(60, 7)
            
            profile = UserProfile(
                user_id=user_id,
                age=age,
                gender=gender,
                height_cm=max(150, min(200, height)),
                weight_kg=max(45, min(120, weight)),
                vo2_max_baseline=vo2_max,
                weekly_miles_baseline=weekly_miles,
                injury_prone=injury_prone,
                years_running=years_running,
                persona_type=persona
            )
            profiles.append(profile)
            
        return profiles
    
    def generate_training_data(
        self,
        profile: UserProfile,
        days: int = 90,
        start_date: datetime = None
    ) -> List[TrainingData]:
        """Generate training data for a single user over specified days."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        training_data = []
        current_vo2_max = profile.vo2_max_baseline
        
        weekly_pattern = self._get_weekly_pattern(profile.persona_type)
        
        for day in range(days):
            date = start_date + timedelta(days=day)
            day_of_week = date.weekday()
            
            workout_type = weekly_pattern[day_of_week]
            
            if workout_type == "rest":
                distance = 0
                duration = 0
                avg_hr = 60
                max_hr = 60
                pace = 0
                elevation = 0
            else:
                distance, duration, avg_hr, max_hr, pace, elevation = self._generate_workout_metrics(
                    profile, workout_type, current_vo2_max
                )
            
            hrv = self._generate_hrv(profile, workout_type, day)
            sleep = self._generate_sleep(workout_type)
            stress = random.uniform(2, 8) if workout_type != "rest" else random.uniform(1, 5)
            temperature = 20 + 10 * np.sin((day + 180) * 2 * np.pi / 365)
            humidity = random.uniform(30, 80)
            fatigue = self._calculate_fatigue(training_data, workout_type)
            
            if day % 7 == 0 and day > 0:
                current_vo2_max = self._update_vo2_max(
                    current_vo2_max, profile.vo2_max_baseline, training_data[-7:]
                )
            
            training = TrainingData(
                user_id=profile.user_id,
                date=date,
                workout_type=workout_type,
                distance_km=distance,
                duration_min=duration,
                avg_hr=avg_hr,
                max_hr=max_hr,
                avg_pace_min_per_km=pace,
                elevation_gain_m=elevation,
                hrv_morning=hrv,
                sleep_hours=sleep,
                stress_score=stress,
                temperature_c=temperature,
                humidity_pct=humidity,
                fatigue_level=fatigue,
                current_vo2_max=current_vo2_max
            )
            training_data.append(training)
        
        return training_data
    
    def _get_weekly_pattern(self, persona: str) -> List[str]:
        """Get typical weekly workout pattern based on persona."""
        patterns = {
            "elite": ["easy", "intervals", "easy", "tempo", "easy", "long", "recovery"],
            "competitive": ["easy", "intervals", "recovery", "tempo", "easy", "long", "rest"],
            "recreational": ["rest", "easy", "recovery", "easy", "rest", "long", "rest"],
            "beginner": ["rest", "easy", "rest", "easy", "rest", "easy", "rest"]
        }
        return patterns[persona]
    
    def _generate_workout_metrics(
        self,
        profile: UserProfile,
        workout_type: str,
        current_vo2_max: float
    ) -> tuple:
        """Generate realistic workout metrics based on type and fitness."""
        max_hr = 220 - profile.age
        
        if workout_type == "easy":
            distance = random.uniform(5, 15)
            pace = 6.5 - (current_vo2_max - 30) * 0.05
            hr_zone = 0.65
        elif workout_type == "tempo":
            distance = random.uniform(8, 12)
            pace = 5.5 - (current_vo2_max - 30) * 0.04
            hr_zone = 0.80
        elif workout_type == "intervals":
            distance = random.uniform(6, 10)
            pace = 5.0 - (current_vo2_max - 30) * 0.04
            hr_zone = 0.90
        elif workout_type == "long":
            distance = random.uniform(15, 30) if profile.persona_type in ["elite", "competitive"] else random.uniform(10, 20)
            pace = 6.8 - (current_vo2_max - 30) * 0.05
            hr_zone = 0.70
        elif workout_type == "recovery":
            distance = random.uniform(3, 8)
            pace = 7.5 - (current_vo2_max - 30) * 0.03
            hr_zone = 0.60
        else:
            return 0, 0, 60, 60, 0, 0
        
        if profile.persona_type == "beginner":
            distance *= 0.5
            pace += 1.5
        elif profile.persona_type == "recreational":
            distance *= 0.7
            pace += 0.8
        
        duration = distance * pace
        avg_hr = int(max_hr * hr_zone + random.uniform(-5, 5))
        max_hr_workout = int(avg_hr + random.uniform(10, 25))
        elevation = random.uniform(0, 50) * distance if random.random() < 0.3 else 0
        
        pace = max(3.5, min(9.5, pace + random.uniform(-0.3, 0.3)))
        
        return distance, duration, avg_hr, max_hr_workout, pace, elevation
    
    def _generate_hrv(self, profile: UserProfile, workout_type: str, day: int) -> float:
        """Generate realistic HRV values."""
        base_hrv = 50 + (70 - profile.age) * 0.5
        
        if profile.gender == "F":
            base_hrv -= 5
        
        base_hrv += (profile.vo2_max_baseline - 45) * 0.8
        
        if workout_type in ["intervals", "tempo"]:
            hrv_modifier = -10
        elif workout_type == "long":
            hrv_modifier = -5
        elif workout_type == "rest":
            hrv_modifier = 5
        else:
            hrv_modifier = 0
        
        daily_variation = np.sin(day * 2 * np.pi / 7) * 5
        
        hrv = base_hrv + hrv_modifier + daily_variation + random.uniform(-8, 8)
        
        return max(20, min(150, hrv))
    
    def _generate_sleep(self, workout_type: str) -> float:
        """Generate realistic sleep hours based on training."""
        if workout_type in ["long", "intervals", "tempo"]:
            base_sleep = random.uniform(7.5, 9)
        elif workout_type == "rest":
            base_sleep = random.uniform(7, 8.5)
        else:
            base_sleep = random.uniform(6.5, 8)
        
        return max(4, min(11, base_sleep + random.uniform(-0.5, 0.5)))
    
    def _calculate_fatigue(self, recent_training: List[TrainingData], workout_type: str) -> float:
        """Calculate accumulated fatigue based on recent training."""
        if len(recent_training) < 7:
            return random.uniform(3, 5)
        
        recent_load = sum(
            t.distance_km * (t.avg_pace_min_per_km / 6.0)
            for t in recent_training[-7:]
            if t.distance_km > 0
        )
        
        base_fatigue = min(9, recent_load / 15)
        
        if workout_type in ["intervals", "tempo", "long"]:
            base_fatigue += random.uniform(0.5, 2)
        
        return max(1, min(10, base_fatigue + random.uniform(-1, 1)))
    
    def _update_vo2_max(
        self,
        current: float,
        baseline: float,
        recent_workouts: List[TrainingData]
    ) -> float:
        """Update VO2max based on training effect."""
        quality_workouts = sum(
            1 for w in recent_workouts
            if w.workout_type in ["tempo", "intervals"]
        )
        
        total_distance = sum(w.distance_km for w in recent_workouts)
        
        if quality_workouts >= 2 and total_distance > 30:
            improvement = random.uniform(0.1, 0.3)
        elif quality_workouts >= 1 and total_distance > 20:
            improvement = random.uniform(0, 0.2)
        else:
            improvement = random.uniform(-0.1, 0.1)
        
        new_vo2 = current + improvement
        max_improvement = baseline * 1.15
        
        return max(30, max(baseline * 0.95, min(max_improvement, min(70, new_vo2))))

    def save_to_files(
        self,
        profiles: List[UserProfile],
        training_data: Dict[str, List[TrainingData]],
        output_dir: Path
    ):
        """Save generated data to CSV and parquet files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        profiles_df = pd.DataFrame([p.model_dump() for p in profiles])
        profiles_df.to_csv(output_dir / "user_profiles.csv", index=False)
        profiles_df.to_parquet(output_dir / "user_profiles.parquet", index=False)
        
        all_training = []
        for user_id, data in training_data.items():
            all_training.extend([d.model_dump() for d in data])
        
        training_df = pd.DataFrame(all_training)
        training_df.to_csv(output_dir / "training_data.csv", index=False)
        training_df.to_parquet(output_dir / "training_data.parquet", index=False)
        
        print(f"✓ Generated {len(profiles)} user profiles")
        print(f"✓ Generated {len(all_training)} training records")
        print(f"✓ Saved to {output_dir}")


def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("--users", type=int, default=100, help="Number of users to generate")
    parser.add_argument("--days", type=int, default=90, help="Number of training days per user")
    parser.add_argument("--output", type=str, default="data/sample_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(seed=args.seed)
    
    print(f"Generating data for {args.users} users over {args.days} days...")
    
    profiles = generator.generate_user_profiles(num_users=args.users)
    
    training_data = {}
    for i, profile in enumerate(profiles):
        if (i + 1) % 10 == 0:
            print(f"  Processing user {i + 1}/{args.users}")
        training_data[profile.user_id] = generator.generate_training_data(profile, days=args.days)
    
    generator.save_to_files(profiles, training_data, Path(args.output))


if __name__ == "__main__":
    main()