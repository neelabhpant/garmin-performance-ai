"""Garmin AI Coach - WHOOP-style conversational coaching chatbot."""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import openai
from pathlib import Path
from dotenv import load_dotenv


class GarminCoach:
    """AI-powered conversational coach similar to WHOOP Coach."""
    
    def __init__(self, predictions: Dict, user_profile: Optional[Dict] = None):
        """Initialize the Garmin Coach with user context.
        
        Args:
            predictions: Current predictions and metrics from the model
            user_profile: Optional user profile information
        """
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = "gpt-4o"
        
        # Store user context
        self.predictions = predictions
        self.user_profile = user_profile or {}
        
        # Build initial context
        self.context = self._build_context()
        
    def _build_context(self) -> str:
        """Build comprehensive context from predictions and profile."""
        context_parts = []
        
        # Current fitness metrics
        context_parts.append(f"Current Fitness Status:")
        context_parts.append(f"• Readiness: {self.predictions.get('readiness_score', 70):.0f}%")
        context_parts.append(f"• VO2 Max: {self.predictions.get('recent_vo2_max', 45):.1f} ml/kg/min")
        context_parts.append(f"• Weekly Mileage: {self.predictions.get('weekly_mileage', 25):.0f} miles")
        context_parts.append(f"• Injury Risk: {self.predictions.get('injury_risk_level', 'Low')} ({self.predictions.get('injury_risk_pct', 20):.0f}%)")
        
        # Race predictions
        context_parts.append(f"\nRace Predictions:")
        context_parts.append(f"• 5K: {self.predictions.get('race_5k_formatted', '25:00')}")
        context_parts.append(f"• 10K: {self.predictions.get('race_10k_formatted', '52:00')}")
        context_parts.append(f"• Half Marathon: {self.predictions.get('race_half_formatted', '2:00:00')}")
        context_parts.append(f"• Marathon: {self.predictions.get('race_marathon_formatted', '4:30:00')}")
        
        # Training metrics
        context_parts.append(f"\nTraining Metrics:")
        context_parts.append(f"• ACWR: {self.predictions.get('acwr', 1.0):.2f}")
        context_parts.append(f"• Recovery Score: {self.predictions.get('recovery_score', 70):.0f}%")
        context_parts.append(f"• Sleep Average: {self.predictions.get('sleep_avg', 7.5):.1f} hours")
        context_parts.append(f"• Recent Pace: {self.predictions.get('avg_pace_recent', 6.0):.1f} min/km")
        
        # Trends
        context_parts.append(f"\nTrends:")
        vo2_trend = self.predictions.get('vo2_max_trend', 0.5)
        trend_direction = "improving" if vo2_trend > 0 else "declining" if vo2_trend < 0 else "stable"
        context_parts.append(f"• VO2 Max Trend: {trend_direction} ({vo2_trend:+.1f}/month)")
        
        return "\n".join(context_parts)
    
    def chat(self, message: str, chat_history: List[Dict] = None) -> str:
        """Process a chat message and return coach response.
        
        Args:
            message: User's message/question
            chat_history: Previous conversation history
            
        Returns:
            Coach's response
        """
        chat_history = chat_history or []
        
        # Build conversation messages
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt()
            }
        ]
        
        # Add conversation history (last 5 exchanges)
        for entry in chat_history[-10:]:  # Last 5 exchanges (user + assistant)
            messages.append(entry)
        
        # Add current user message
        messages.append({"role": "user", "content": message})
        
        try:
            # Get response from GPT-4
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=400,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback response if API fails
            return self._get_fallback_response(message)
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the coach personality."""
        return f"""You are a supportive, knowledgeable running coach having a conversation with your athlete. 
        You have access to their real-time biometric and performance data.
        
        {self.context}
        
        Personality traits:
        - Encouraging and motivating, but realistic
        - Use "you/your" to speak directly to the athlete
        - Provide specific, actionable advice based on their data
        - Keep responses concise (2-3 paragraphs max)
        - Ask follow-up questions to engage them
        - Balance pushing performance with injury prevention
        - Reference their specific metrics when giving advice
        
        Remember:
        - If readiness is below 60%, prioritize recovery
        - If injury risk is high, be cautious with intensity recommendations
        - Use their actual predicted race times when discussing goals
        - Consider their ACWR when suggesting training changes"""
    
    def _get_fallback_response(self, message: str) -> str:
        """Provide a fallback response if API fails."""
        message_lower = message.lower()
        
        # Readiness-based responses
        readiness = self.predictions.get('readiness_score', 70)
        
        if any(word in message_lower for word in ['today', 'now', 'should i run', 'workout']):
            if readiness > 80:
                return f"With your readiness at {readiness:.0f}%, you're primed for a quality session today! Consider a tempo run or intervals. Your body is responding well to training. What type of workout sounds good to you?"
            elif readiness > 60:
                return f"Your readiness is at {readiness:.0f}%, which suggests moderate intensity today. I'd recommend a steady run at your easy pace, focusing on good form. How did you sleep last night?"
            else:
                return f"Your readiness is only {readiness:.0f}% today. This is your body asking for recovery. Consider a light 20-30 minute recovery run or take the day off. Rest is when adaptation happens. How are you feeling?"
        
        elif any(word in message_lower for word in ['race', 'marathon', '5k', '10k']):
            return f"Based on your current fitness, you're looking at a {self.predictions.get('race_marathon_formatted', '4:30:00')} marathon and {self.predictions.get('race_5k_formatted', '25:00')} for 5K. Your VO2 max of {self.predictions.get('recent_vo2_max', 45):.1f} shows solid aerobic fitness. Which race distance are you most interested in?"
        
        elif any(word in message_lower for word in ['injury', 'hurt', 'pain', 'sore']):
            risk = self.predictions.get('injury_risk_pct', 20)
            return f"Your injury risk is currently {risk:.0f}%, which is {self.predictions.get('injury_risk_level', 'low')}. If you're experiencing pain, don't push through it. Consider taking a rest day or doing some cross-training. Where specifically are you feeling discomfort?"
        
        else:
            return f"Based on your current metrics, you're doing well! Your readiness is {readiness:.0f}% and your training appears balanced. What specific aspect of your training would you like to discuss?"
    
    def get_suggested_questions(self) -> List[str]:
        """Get contextually relevant question suggestions."""
        suggestions = []
        
        readiness = self.predictions.get('readiness_score', 70)
        injury_risk = self.predictions.get('injury_risk_pct', 20)
        
        # Always include
        suggestions.append("What should I do for today's workout?")
        
        # Readiness-based
        if readiness < 60:
            suggestions.append("Why is my readiness so low?")
            suggestions.append("Should I take a rest day?")
        else:
            suggestions.append("Am I ready for a hard workout?")
        
        # Goal-based
        suggestions.append("How can I improve my 5K time?")
        suggestions.append("What's my marathon potential?")
        
        # Recovery/Injury
        if injury_risk > 30:
            suggestions.append("How can I reduce my injury risk?")
        
        suggestions.append("How important is sleep for performance?")
        
        # Training
        suggestions.append("Should I increase my weekly mileage?")
        suggestions.append("What pace should I run my long runs?")
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def format_response_with_data(self, response: str) -> Tuple[str, Optional[Dict]]:
        """Format response and extract any data points mentioned.
        
        Returns:
            Tuple of (formatted_response, data_highlights)
        """
        # Highlight any metrics mentioned in response
        data_highlights = {}
        
        # Check if response mentions specific metrics
        if "readiness" in response.lower():
            data_highlights["Readiness"] = f"{self.predictions.get('readiness_score', 70):.0f}%"
        
        if "vo2" in response.lower():
            data_highlights["VO2 Max"] = f"{self.predictions.get('recent_vo2_max', 45):.1f}"
        
        if "injury" in response.lower():
            data_highlights["Injury Risk"] = self.predictions.get('injury_risk_level', 'Low')
        
        return response, data_highlights if data_highlights else None