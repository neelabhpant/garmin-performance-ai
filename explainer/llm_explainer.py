"""GPT-4o LLM integration for intelligent explanations and recommendations."""

import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib
import json
from pathlib import Path

import openai
from dotenv import load_dotenv

from .prompts import (
    RACE_PREDICTION_PROMPT,
    TRAINING_RECOMMENDATION_PROMPT,
    INJURY_RISK_PROMPT,
    TREND_ANALYSIS_PROMPT,
    COMPREHENSIVE_COACH_PROMPT,
    CHAT_COACH_PROMPT,
    WEEKLY_PLAN_PROMPT
)


class LLMExplainer:
    """Integrates GPT-4o for intelligent coaching insights and explanations."""
    
    def __init__(self, use_cache: bool = True, cache_dir: Optional[Path] = None):
        """Initialize the LLM Explainer.
        
        Args:
            use_cache: Whether to cache responses to reduce API calls
            cache_dir: Directory for cache storage (default: .cache/)
        """
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Model configuration
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = int(os.getenv("OPENAI_MAX_TOKENS", "500"))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        
        # Cache configuration
        self.use_cache = use_cache and os.getenv("USE_LLM_CACHE", "true").lower() == "true"
        self.cache_dir = cache_dir or Path(".cache/llm_responses")
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, prompt: str, context: Dict) -> str:
        """Generate a cache key from prompt and context."""
        cache_data = f"{prompt}{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Retrieve cached response if available."""
        if not self.use_cache:
            return None
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is less than 24 hours old
                    if (datetime.now() - datetime.fromisoformat(data['timestamp'])).days < 1:
                        return data['response']
            except Exception:
                pass
        return None
    
    def _save_to_cache(self, cache_key: str, response: str):
        """Save response to cache."""
        if not self.use_cache:
            return
        
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'response': response,
                'timestamp': datetime.now().isoformat()
            }, f)
    
    def _call_gpt(self, prompt: str, context: Dict) -> str:
        """Make API call to GPT-4o."""
        # Check cache first
        cache_key = self._get_cache_key(prompt, context)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached
        
        # Format prompt with context
        formatted_prompt = prompt.format(**context)
        
        try:
            # Log the API call attempt
            print(f"Making OpenAI API call with model: {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a friendly, knowledgeable running coach who cares deeply about your athlete's success. Speak directly to them, be encouraging, and provide specific, actionable advice."},
                    {"role": "user", "content": formatted_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            result = response.choices[0].message.content
            
            # Save to cache
            self._save_to_cache(cache_key, result)
            
            print(f"OpenAI API call successful, response length: {len(result)} chars")
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"OpenAI API Error: {error_msg}")
            
            # Provide more specific error messages
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return "🔑 API key issue detected. Please check your OpenAI API key in the .env file."
            elif "rate" in error_msg.lower():
                return "⏳ API rate limit reached. Please try again in a few moments."
            elif "model" in error_msg.lower():
                return f"🤖 Model '{self.model}' not available. Check your OpenAI subscription."
            else:
                return f"💭 Having trouble connecting to AI coach. Let's use standard analysis for now."
    
    def explain_race_prediction(self, predictions: Dict, features: Dict) -> str:
        """Generate personalized race strategy explanation.
        
        Args:
            predictions: Model predictions including race times
            features: User features including training metrics
            
        Returns:
            Personalized race strategy and training advice
        """
        context = {
            'race_5k': predictions.get('race_5k_formatted', 'N/A'),
            'race_10k': predictions.get('race_10k_formatted', 'N/A'),
            'race_half': predictions.get('race_half_formatted', 'N/A'),
            'race_marathon': predictions.get('race_marathon_formatted', 'N/A'),
            'vo2_max': features.get('recent_vo2_max', 45),
            'weekly_miles': features.get('weekly_mileage', 25),
            'avg_pace': features.get('avg_pace_recent', 6.0),
            'acwr': features.get('acwr', 1.0)
        }
        
        return self._call_gpt(RACE_PREDICTION_PROMPT, context)
    
    def generate_training_recommendation(self, readiness: int, status: str, features: Dict) -> str:
        """Create dynamic training plan based on readiness.
        
        Args:
            readiness: Readiness score (0-100)
            status: Status level (green/yellow/red)
            features: User training features
            
        Returns:
            Personalized workout recommendation
        """
        context = {
            'readiness': readiness,
            'status': status,
            'acute_load': features.get('acute_load', 40),
            'chronic_load': features.get('chronic_load', 35),
            'acwr': features.get('acwr', 1.0),
            'hrv_trend': 'stable',  # Would calculate from data
            'sleep_avg': features.get('sleep_avg', 7.5),
            'recovery_score': features.get('recovery_score', 70),
            'vo2_max': features.get('recent_vo2_max', 45)
        }
        
        return self._call_gpt(TRAINING_RECOMMENDATION_PROMPT, context)
    
    def explain_injury_risk(self, risk_level: str, risk_pct: float, features: Dict) -> str:
        """Provide detailed injury prevention guidance.
        
        Args:
            risk_level: Low/Moderate/High
            risk_pct: Risk percentage
            features: User features including risk factors
            
        Returns:
            Prevention strategies and exercises
        """
        context = {
            'risk_level': risk_level,
            'risk_pct': risk_pct,
            'acwr': features.get('acwr', 1.0),
            'fatigue_level': features.get('fatigue_accumulated', 5),
            'mileage_trend': features.get('mileage_trend', 0),
            'sleep_avg': features.get('sleep_avg', 7.5),
            'recovery_score': features.get('recovery_score', 70),
            'years_running': features.get('years_running', 5),
            'injury_history': features.get('injury_history', 0)
        }
        
        return self._call_gpt(INJURY_RISK_PROMPT, context)
    
    def analyze_performance_trends(self, trend_data: Dict, features: Dict) -> str:
        """Provide narrative explanation of performance trajectory.
        
        Args:
            trend_data: Trend metrics including VO2 max changes
            features: User training features
            
        Returns:
            Performance analysis and recommendations
        """
        context = {
            'vo2_current': trend_data.get('vo2_current', 45),
            'vo2_trend': trend_data.get('vo2_trend', 0.1),
            'vo2_projection': trend_data.get('vo2_projection', 45.5),
            'mileage_trend': features.get('mileage_trend', 0),
            'pace_improvement': features.get('pace_improvement', 0),
            'recent_5k': 'N/A',  # Would get from recent performances
            'consistency': 85  # Would calculate from training data
        }
        
        return self._call_gpt(TREND_ANALYSIS_PROMPT, context)
    
    def generate_comprehensive_report(self, predictions: Dict, features: Dict) -> str:
        """Generate complete coaching report combining all metrics.
        
        Args:
            predictions: All model predictions
            features: Complete user feature set
            
        Returns:
            Comprehensive coaching report
        """
        context = {
            'race_5k': predictions.get('race_5k_formatted', 'N/A'),
            'race_10k': predictions.get('race_10k_formatted', 'N/A'),
            'race_half': predictions.get('race_half_formatted', 'N/A'),
            'race_marathon': predictions.get('race_marathon_formatted', 'N/A'),
            'vo2_max': features.get('recent_vo2_max', 45),
            'vo2_trend': features.get('vo2_max_trend', 0.1),
            'readiness': predictions.get('readiness_score', 70),
            'injury_risk_level': predictions.get('injury_risk_level', 'Moderate'),
            'injury_risk_pct': predictions.get('injury_risk_pct', 20),
            'weekly_miles': features.get('weekly_mileage', 25),
            'acwr': features.get('acwr', 1.0),
            'recovery_score': features.get('recovery_score', 70),
            'consistency': 85,
            'years_running': features.get('years_running', 5),
            'recent_focus': 'Base building'  # Would determine from training data
        }
        
        # Use higher token limit for comprehensive report
        original_max = self.max_tokens
        self.max_tokens = 800
        result = self._call_gpt(COMPREHENSIVE_COACH_PROMPT, context)
        self.max_tokens = original_max
        
        return result
    
    def chat_with_coach(self, question: str, context: Dict) -> str:
        """Interactive chat with AI coach.
        
        Args:
            question: User's question
            context: Current user state and history
            
        Returns:
            Coach's response
        """
        chat_context = {
            'fitness_summary': f"VO2max {context['features'].get('recent_vo2_max', 45):.1f}, {context['features'].get('weekly_mileage', 25):.1f} miles/week",
            'recent_performance': f"5K: {context['predictions'].get('race_5k_formatted', 'N/A')}",
            'goals': 'Improve race times',  # Would get from user profile
            'readiness': context['predictions'].get('readiness_score', 70),
            'risk_factors': context['predictions'].get('injury_risk_level', 'Moderate'),
            'chat_history': '\n'.join([f"User: {h['user']}\nCoach: {h['coach']}" for h in context.get('history', [])]),
            'question': question
        }
        
        return self._call_gpt(CHAT_COACH_PROMPT, chat_context)
    
    def generate_weekly_plan(self, readiness: int, goals: Dict, features: Dict) -> str:
        """Generate detailed weekly training plan.
        
        Args:
            readiness: Current readiness score
            goals: User's training goals
            features: User features
            
        Returns:
            Detailed 7-day training plan
        """
        context = {
            'vo2_max': features.get('recent_vo2_max', 45),
            'weekly_miles': features.get('weekly_mileage', 25),
            'goal_race': goals.get('race_distance', 'Half Marathon'),
            'weeks_to_race': goals.get('weeks_to_race', 12),
            'readiness': readiness,
            'injury_risk': 'Low',  # Would calculate
            'available_hours': goals.get('hours_per_week', 7),
            'weaknesses': 'Speed endurance'  # Would determine from data
        }
        
        # Use higher token limit for weekly plan
        original_max = self.max_tokens
        self.max_tokens = 600
        result = self._call_gpt(WEEKLY_PLAN_PROMPT, context)
        self.max_tokens = original_max
        
        return result