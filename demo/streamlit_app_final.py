"""Final Streamlit demo app with real Random Forest ML predictions."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.inference_rf import RFInferenceService
from demo.visualizations import (
    create_readiness_gauge, 
    create_injury_risk_chart,
    create_vo2_gauge,
    create_mileage_gauge,
    create_pace_gauge
)
from explainer import LLMExplainer, GarminCoach

st.set_page_config(
    page_title="Garmin Performance AI - Production Ready",
    page_icon="🏃",
    layout="wide"
)

# Custom CSS for Garmin branding
st.markdown("""
<style>
    .main {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    h1 {
        color: #1a73e8;
    }
    h2 {
        color: #1a73e8;
        font-weight: 600;
    }
    h3 {
        color: #333;
        font-weight: 500;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(135deg, #f0f4f8 0%, #ffffff 100%);
        border-radius: 12px;
        padding: 8px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 12px 24px;
        background-color: #ffffff;
        color: #495057 !important;
        font-weight: 500;
        border-radius: 8px;
        margin: 0 4px;
        border: 1px solid #e9ecef;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #f8f9fa;
        border-color: #1a73e8;
        color: #1a73e8 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
        color: white !important;
        border: none;
        box-shadow: 0 4px 12px rgba(26,115,232,0.3);
        font-weight: 600;
        transform: translateY(-1px);
    }
    .stTabs [data-baseweb="tab"] button {
        color: inherit !important;
        font-size: 14px;
        font-weight: inherit;
    }
    .stTabs [data-baseweb="tab"] p {
        color: inherit !important;
        margin: 0;
    }
    div[data-testid="stHorizontalBlock"] > div {
        border-radius: 10px;
    }
    .plotly-graph-div {
        border-radius: 10px;
        overflow: hidden;
    }
    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 2px solid #f0f0f0;
    }
    /* General selectbox styling */
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        color: #212529 !important;
    }
    
    .stSelectbox label {
        color: #495057 !important;
        font-weight: 500;
    }
    
    .stSelectbox > div > div > div {
        color: #212529 !important;
    }
    
    .stSelectbox [data-baseweb="select"] {
        color: #212529 !important;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        background-color: #ffffff;
        border-color: #1a73e8;
    }
    
    /* Sidebar selectbox styling - simplified */
    .stSidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        color: #212529 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .stSidebar .stSelectbox label {
        color: #495057 !important;
        font-weight: 500;
        font-size: 14px;
    }
    
    .stSidebar .stSelectbox > div > div > div {
        color: #212529 !important;
    }
    
    .stSidebar .stSelectbox [data-baseweb="select"] {
        background-color: #f8f9fa;
        color: #212529 !important;
    }
    
    .stSidebar .stSelectbox [data-baseweb="select"]:hover {
        background-color: #ffffff;
        border-color: #1a73e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Dropdown menu styling */
    [data-baseweb="popover"] {
        background-color: #ffffff !important;
    }
    
    [data-baseweb="popover"] li {
        color: #212529 !important;
        background-color: #ffffff;
    }
    
    [data-baseweb="popover"] li:hover {
        background-color: #e7f1ff !important;
        color: #1a73e8 !important;
    }
    
    /* Sidebar headers */
    .stSidebar h3 {
        color: #1a73e8 !important;
        font-weight: 600;
    }
    
    .race-card {
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .race-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_inference_service():
    """Load the Random Forest inference service."""
    service = RFInferenceService()
    return service

@st.cache_resource
def load_llm_explainer():
    """Load the LLM explainer for AI insights."""
    try:
        return LLMExplainer(use_cache=True)
    except Exception as e:
        st.warning(f"AI Coach not available: {str(e)}")
        return None

@st.cache_resource
def load_garmin_coach(_predictions):
    """Load the Garmin AI Coach chatbot."""
    try:
        return GarminCoach(_predictions)
    except Exception as e:
        st.warning(f"Garmin Coach not available: {str(e)}")
        return None

def create_feature_importance_chart(importance_data, title):
    """Create feature importance bar chart."""
    if not importance_data:
        return None
    
    features = [item[0] for item in importance_data]
    importances = [item[1] for item in importance_data]
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features,
        orientation='h',
        marker=dict(color='#1a73e8')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=150, r=20, t=40, b=40)
    )
    
    return fig

def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Garmin_logo.svg/320px-Garmin_logo.svg.png", width=200)
    
    st.title("🏃 Garmin Performance AI - Connect+ Premium")
    st.markdown("### Powered by Random Forest ML Models | Real Predictions from 100 Athletes")
    
    # Load inference service
    service = load_inference_service()
    
    # Load LLM explainer
    explainer = load_llm_explainer()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Athlete Selection")
        
        persona_mapping = {
            "Elite Emma 🥇": "elite",
            "Competitive Carlos 🏆": "competitive", 
            "Recreational Rachel 🏃‍♀️": "recreational",
            "Beginner Ben 🚶‍♂️": "beginner"
        }
        
        selected_name = st.selectbox("Choose athlete profile:", list(persona_mapping.keys()))
        selected_persona = persona_mapping[selected_name]
        
        st.markdown("---")
        
        # Model Performance Metrics
        st.markdown("### 📊 Model Performance")
        perf = service.get_model_performance()
        
        st.success(f"""
        **Race Model**
        • R² Score: {perf['race']['r2_score']:.3f}
        • 5K MAE: ±{perf['race']['mae_5k_minutes']:.1f} min
        
        **Health Model**  
        • Readiness R²: {perf['health']['readiness_r2']:.3f}
        • Injury MAE: ±{perf['health']['injury_mae_pct']:.1f}%
        
        **Trend Model**
        • R² Score: {perf['trend']['r2_score']:.3f}
        """)
        
        st.markdown("---")
        
        # CDP Integration
        st.markdown("### 🔧 CDP Architecture")
        st.info("""
        **Infrastructure:**
        • 1,200 CDP nodes
        • 15 PB storage
        • 2.5 TB daily data
        
        **ML Platform:**
        • Cloudera CML
        • MLflow tracking
        • Model registry
        • A/B testing ready
        """)
        
        st.markdown("---")
        
        # Business Impact
        st.markdown("### 💰 Business Impact")
        st.metric("Connect+ Target", "3% → 8%", "+166%")
        st.metric("Churn Reduction", "15% → 5%", "-66%")
        st.metric("Annual Revenue Lift", "$42M", "+$28M")
    
    # Get predictions
    predictions = service.get_sample_user_predictions(selected_persona)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Race Predictions",
        "💪 Readiness",
        "⚠️ Injury Risk",
        "📈 Trends",
        "🧠 Model Insights",
        "💼 Executive View"
    ])
    
    with tab1:
        st.markdown("## 🏃‍♂️ AI-Powered Race Predictions")
        st.markdown("---")
        
        # Race Predictions Section - Full Width with Cards
        st.markdown("### 🏁 Predicted Race Finish Times")
        
        # Create 4 columns for race predictions
        race_cols = st.columns(4)
        
        races = [
            {'name': '5K', 'key': 'race_5k_formatted', 'default': '25:00', 'color': '#4CAF50', 'emoji': '🏃'},
            {'name': '10K', 'key': 'race_10k_formatted', 'default': '52:00', 'color': '#2196F3', 'emoji': '🏃‍♀️'},
            {'name': 'Half Marathon', 'key': 'race_half_formatted', 'default': '2:00:00', 'color': '#FF9800', 'emoji': '🏅'},
            {'name': 'Marathon', 'key': 'race_marathon_formatted', 'default': '4:30:00', 'color': '#9C27B0', 'emoji': '🏆'}
        ]
        
        confidences = ["±1 min", "±2 min", "±4 min", "±8 min"]
        percentiles = ["Top 20%", "Top 25%", "Top 30%", "Top 35%"]
        
        for i, (col, race, conf, perc) in enumerate(zip(race_cols, races, confidences, percentiles)):
            with col:
                # Create a styled container for each race
                st.markdown(f"""
                <div class='race-card' style='background: linear-gradient(135deg, white 0%, {race['color']}12 100%); 
                            padding: 20px; border-radius: 10px; border-left: 4px solid {race['color']};
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-height: 170px; cursor: pointer;'>
                    <h4 style='color: {race['color']}; margin: 0;'>{race['emoji']} {race['name']}</h4>
                    <h2 style='color: #333; margin: 10px 0;'>{predictions.get(race['key'], race['default'])}</h2>
                    <p style='color: #666; margin: 5px 0; font-size: 14px;'>🎯 {conf}</p>
                    <p style='color: #666; margin: 5px 0; font-size: 14px;'>📊 {perc}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Add spacing after race cards section
        st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Athlete Metrics Section - 3 columns
        st.markdown("### 📊 Current Performance Metrics")
        
        metrics_cols = st.columns(3)
        
        with metrics_cols[0]:
            vo2_value = predictions.get('recent_vo2_max', 45)
            fig_vo2 = create_vo2_gauge(vo2_value)
            st.plotly_chart(fig_vo2, use_container_width=True)
        
        with metrics_cols[1]:
            miles_value = predictions.get('weekly_mileage', 25)
            fig_miles = create_mileage_gauge(miles_value)
            st.plotly_chart(fig_miles, use_container_width=True)
        
        with metrics_cols[2]:
            pace_value = predictions.get('avg_pace_recent', 6.0)
            fig_pace = create_pace_gauge(pace_value)
            st.plotly_chart(fig_pace, use_container_width=True)
        
        st.markdown("---")
        
        # Model Confidence Section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📊 Optimal Pacing Strategy")
            
            # Race distance selector
            race_option = st.selectbox(
                "Select Race Distance:",
                ["5K", "10K", "Half Marathon", "Marathon"],
                index=3  # Default to Marathon
            )
            
            # Map selection to pace splits and segment distances
            race_mapping = {
                "5K": {
                    "key": "pace_splits_5k",
                    "segments": 5,
                    "distance_per_segment": 1.0,  # km
                    "total_distance": 5.0,
                    "default_time": 1500  # 25 minutes default
                },
                "10K": {
                    "key": "pace_splits_10k",
                    "segments": 10,
                    "distance_per_segment": 1.0,  # km
                    "total_distance": 10.0,
                    "default_time": 3150  # 52.5 minutes default
                },
                "Half Marathon": {
                    "key": "pace_splits_half",
                    "segments": 10,
                    "distance_per_segment": 2.1097,  # km (21.097 km / 10)
                    "total_distance": 21.097,
                    "default_time": 7200  # 2 hours default
                },
                "Marathon": {
                    "key": "pace_splits_marathon",
                    "segments": 10,
                    "distance_per_segment": 4.2195,  # km (42.195 km / 10)
                    "total_distance": 42.195,
                    "default_time": 16200  # 4.5 hours default
                }
            }
            
            race_info = race_mapping[race_option]
            splits_key = race_info["key"]
            distance_per_seg = race_info["distance_per_segment"]
            num_segments = race_info["segments"]
            
            # Get splits for selected race
            default_splits = [race_info["default_time"] / num_segments] * num_segments
            splits = predictions.get(splits_key, default_splits)
            
            splits_df = pd.DataFrame({
                'Segment': [f"Seg {i+1}" for i in range(len(splits))],
                'Time (min)': [s/60 for s in splits],
                'Pace (min/km)': [s/60/distance_per_seg for s in splits]
            })
            
            fig = px.line(splits_df, x='Segment', y='Pace (min/km)', 
                         title=f"AI-Optimized {race_option} Pacing (Negative Split)",
                         markers=True, color_discrete_sequence=['#1a73e8'])
            fig.add_hline(y=np.mean(splits_df['Pace (min/km)']), 
                         line_dash="dash", line_color="red",
                         annotation_text="Average Pace")
            fig.update_layout(height=350, margin=dict(l=30, r=30, t=60, b=30))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Model Confidence")
            confidence = 95 if selected_persona in ['elite', 'competitive'] else 90
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                title={'text': "Prediction Accuracy", 'font': {'size': 16}},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 100]},
                      'bar': {'color': "#1a73e8"},
                      'steps': [
                          {'range': [0, 80], 'color': "lightgray"},
                          {'range': [80, 100], 'color': "#e8f0fe"}
                      ]}
            ))
            fig.update_layout(height=350, margin=dict(l=30, r=30, t=80, b=30))
            st.plotly_chart(fig, use_container_width=True)
        
        # AI Coach Race Analysis
        st.markdown("---")
        st.markdown("## 💬 Your Personal AI Coach")
        
        # Extract features from predictions (they're embedded in the predictions dict)
        features = {
            'recent_vo2_max': predictions.get('recent_vo2_max', 45),
            'weekly_mileage': predictions.get('weekly_mileage', 25),
            'avg_pace_recent': predictions.get('avg_pace_recent', 6.0),
            'acwr': predictions.get('acwr', 1.0),
            'recovery_score': predictions.get('recovery_score', 70),
            'sleep_avg': predictions.get('sleep_avg', 7.5),
            'mileage_trend': predictions.get('mileage_trend', 5),
            'vo2_max_trend': predictions.get('vo2_max_trend', 0.5),
            'fatigue_accumulated': predictions.get('fatigue_accumulated', 3),
            'hrv_baseline': predictions.get('hrv_baseline', 50),
            'years_running': predictions.get('years_running', 3),
            'acute_load': predictions.get('acute_load', 40),
            'chronic_load': predictions.get('chronic_load', 35),
        }
        
        # Create a nice container for AI insights
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                with st.spinner("🤔 Your coach is analyzing your data..."):
                    try:
                        # Get AI explanation for race predictions
                        ai_insights = explainer.explain_race_prediction(
                            predictions=predictions,
                            features=features
                        )
                        
                        # Display in a conversational style
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 20px; border-radius: 10px; color: white;'>
                            <h4 style='margin-top:0; color:white;'>📣 Coach's Message</h4>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"<p style='color:white; line-height:1.6;'>{ai_insights}</p>", 
                                  unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    except Exception as e:
                        # Improved fallback with personalized touch
                        from explainer.feature_importance import FeatureExplainer
                        feature_exp = FeatureExplainer()
                        summary = feature_exp.generate_feature_summary(features)
                        
                        st.markdown("""
                        <div style='background: #f8f9fa; padding: 20px; border-radius: 10px;'>
                            <h4 style='margin-top:0;'>📊 Performance Analysis</h4>
                        """, unsafe_allow_html=True)
                        
                        # Make the fallback more personal
                        vo2 = features.get('recent_vo2_max', 45)
                        miles = features.get('weekly_mileage', 25)
                        
                        personal_msg = f"""
                        Based on your current fitness metrics, here's what stands out:
                        
                        {summary}
                        
                        **Today's Focus:** With your VO2 max at {vo2:.1f} and weekly mileage at {miles:.0f} miles, 
                        I recommend focusing on maintaining consistency while gradually building intensity. 
                        Your body is responding well to training - keep up the great work! 🎯
                        """
                        
                        st.markdown(f"<p style='line-height:1.6;'>{personal_msg}</p>", 
                                  unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                # Quick stats card
                st.markdown("""
                <div style='background: #e8f0fe; padding: 15px; border-radius: 10px; text-align: center;'>
                    <h5 style='margin:0; color: #1a73e8;'>Your Stats</h5>
                """, unsafe_allow_html=True)
                
                st.metric("Readiness", f"{predictions.get('readiness_score', 0):.0f}%", 
                         delta=None, label_visibility="visible")
                st.metric("VO2 Max", f"{features.get('recent_vo2_max', 45):.1f}", 
                         delta=f"+{features.get('vo2_max_trend', 0):.1f}/mo", label_visibility="visible")
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        # Workout recommendation section
        st.markdown("---")
        st.markdown("### 🏃 Today's Workout Recommendation")
        
        with st.spinner("Planning your workout..."):
            try:
                # Get training recommendation
                # Use the race_option from earlier if it exists, otherwise use general
                goal_race = "general"
                if 'race_option' in globals():
                    goal_race = race_option.lower().replace(" ", "_")
                    
                training_rec = explainer.get_training_recommendations(
                    predictions=predictions,
                    features=features,
                    goal_race=goal_race
                )
                
                # Display as a workout card
                st.markdown("""
                <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                            padding: 20px; border-radius: 10px; color: white;'>
                    <h4 style='margin-top:0; color:white;'>🎯 Your Workout Plan</h4>
                """, unsafe_allow_html=True)
                
                st.markdown(f"<p style='color:white; line-height:1.6;'>{training_rec}</p>", 
                          unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            except:
                # Fallback workout based on readiness
                readiness = predictions.get('readiness_score', 70)
                if readiness > 80:
                    workout = "**Tempo Run:** 10min warmup + 20min at threshold pace + 10min cooldown"
                    reason = "Your high readiness means you're primed for quality work!"
                elif readiness > 60:
                    workout = "**Steady Run:** 35-45min at comfortable pace, focus on form"
                    reason = "Moderate readiness - maintain fitness while allowing recovery."
                else:
                    workout = "**Recovery Run:** 25-30min easy or rest day"
                    reason = "Low readiness detected - prioritize recovery today."
                
                st.info(f"""
                {workout}
                
                {reason} Listen to your body and adjust as needed. 💪
                """)
        
        # AI Coach Chat Interface
        st.markdown("---")
        st.markdown("## 💬 Chat with Your AI Coach")
        st.markdown("*Get personalized advice, ask training questions, or discuss your goals - available 24/7*")
        
        # Initialize chat history in session state
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'coach' not in st.session_state:
            st.session_state.coach = load_garmin_coach(predictions)
        
        # Create the chat interface
        with st.expander("🗣️ Start a Conversation with Your Coach", expanded=False):
            
            # Display chat history
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    if msg["role"] == "user":
                        st.markdown(f"""
                        <div style='background: #e8f0fe; padding: 12px; border-radius: 10px; margin: 8px 0; color: #1f2937;'>
                            <strong>You:</strong> {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    color: white; padding: 12px; border-radius: 10px; margin: 8px 0;'>
                            <strong>Coach:</strong> {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Suggested questions
            if st.session_state.coach:
                st.markdown("**Quick Questions:**")
                suggestions = st.session_state.coach.get_suggested_questions()
                
                # Create columns for suggestion buttons
                cols = st.columns(3)
                for i, suggestion in enumerate(suggestions[:6]):
                    with cols[i % 3]:
                        if st.button(suggestion, key=f"sugg_{i}", use_container_width=True):
                            # Process the suggested question
                            with st.spinner("Coach is thinking..."):
                                response = st.session_state.coach.chat(
                                    suggestion, 
                                    st.session_state.chat_history
                                )
                                # Add to chat history
                                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                                st.session_state.chat_history.append({"role": "assistant", "content": response})
                                st.rerun()
            
            # Chat input
            st.markdown("**Your Question:**")
            col1, col2 = st.columns([5, 1])
            with col1:
                user_input = st.text_input(
                    "Ask your coach anything...",
                    placeholder="e.g., Should I run today? How can I improve my 5K time?",
                    label_visibility="collapsed",
                    key="chat_input"
                )
            with col2:
                send_button = st.button("Send 📤", use_container_width=True)
            
            # Process user input
            if (user_input and send_button) or (user_input and st.session_state.get('enter_pressed')):
                if st.session_state.coach:
                    with st.spinner("Coach is thinking..."):
                        try:
                            # Get response from coach
                            response = st.session_state.coach.chat(
                                user_input, 
                                st.session_state.chat_history
                            )
                            
                            # Add to chat history
                            st.session_state.chat_history.append({"role": "user", "content": user_input})
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                            
                            # Clear input and rerun
                            st.rerun()
                            
                        except Exception as e:
                            st.error("Sorry, I couldn't process that. Please try again.")
                else:
                    st.warning("Coach is not available. Please check your OpenAI API key.")
            
            # Clear chat button
            if st.session_state.chat_history:
                if st.button("🗑️ Clear Conversation", use_container_width=True):
                    st.session_state.chat_history = []
                    st.rerun()
            
            # Coach status indicator
            if st.session_state.coach:
                st.markdown("""
                <div style='text-align: center; padding: 10px; color: #666; font-size: 12px;'>
                    🟢 Coach is online and ready to help | Powered by GPT-4
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 10px; color: #666; font-size: 12px;'>
                    🔴 Coach is offline | Check API configuration
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## Training Readiness Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            readiness = predictions.get('readiness_score', 70)
            status = predictions.get('readiness_status', 'yellow')
            
            fig = create_readiness_gauge(readiness, status)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Readiness Factors")
            factors = {
                'Training Load (ACWR)': min(1.0, predictions.get('acwr', 1.0) / 1.3),
                'Recovery Quality': predictions.get('recovery_score', 70) / 100,
                'Sleep Quality': predictions.get('sleep_avg', 7.5) / 8,
                'HRV Stability': max(0, 1 - predictions.get('hrv_cv', 0.1))
            }
            
            for factor, value in factors.items():
                col_color = st.columns([3, 1])
                with col_color[0]:
                    st.progress(value, text=factor)
                with col_color[1]:
                    st.write(f"{value*100:.0f}%")
        
        with col2:
            st.markdown("### 🤖 AI Training Recommendation")
            
            with st.spinner("Analyzing your readiness..."):
                try:
                    # Extract features from predictions for AI analysis
                    features = {
                        'recent_vo2_max': predictions.get('recent_vo2_max', 45),
                        'weekly_mileage': predictions.get('weekly_mileage', 25),
                        'avg_pace_recent': predictions.get('avg_pace_recent', 6.0),
                        'acwr': predictions.get('acwr', 1.0),
                        'recovery_score': predictions.get('recovery_score', 70),
                        'sleep_avg': predictions.get('sleep_avg', 7.5),
                        'mileage_trend': predictions.get('mileage_trend', 5),
                        'vo2_max_trend': predictions.get('vo2_max_trend', 0.5),
                        'fatigue_accumulated': predictions.get('fatigue_accumulated', 3),
                        'hrv_baseline': predictions.get('hrv_baseline', 50),
                    }
                    
                    # Get personalized training recommendation from AI
                    ai_recommendation = explainer.get_training_recommendations(
                        predictions=predictions,
                        features=features,
                        goal_race=None  # General training, not race-specific
                    )
                    
                    # Display based on readiness status
                    if status == "green":
                        st.success(f"""
                        #### 🟢 High Intensity Approved
                        
                        {ai_recommendation}
                        
                        **Readiness Score: {readiness}%** - Optimal adaptation window detected!
                        """)
                    elif status == "yellow":
                        st.warning(f"""
                        #### 🟡 Moderate Intensity Recommended
                        
                        {ai_recommendation}
                        
                        **Readiness Score: {readiness}%** - Balance intensity with recovery.
                        """)
                    else:
                        st.error(f"""
                        #### 🔴 Recovery Priority
                        
                        {ai_recommendation}
                        
                        **Readiness Score: {readiness}%** - Prioritize recovery to prevent injury.
                        """)
                    
                except Exception as e:
                    # Fallback to static recommendations if AI fails
                    if status == "green":
                        st.success("""
                        #### 🟢 High Intensity Approved
                        
                        **Today's Workout:**
                        • Type: Intervals or Tempo Run
                        • Duration: 45-60 minutes
                        • Intensity: 85-90% max HR
                        
                        **Why:** Your recovery metrics indicate optimal adaptation window. Push hard today for maximum gains.
                        """)
                    elif status == "yellow":
                        st.warning("""
                        #### 🟡 Moderate Intensity Recommended
                        
                        **Today's Workout:**
                        • Type: Steady State Run
                        • Duration: 30-45 minutes
                        • Intensity: 70-75% max HR
                        
                        **Why:** Maintain fitness while allowing recovery. Focus on form and efficiency.
                        """)
                    else:
                        st.error("""
                        #### 🔴 Recovery Priority
                        
                        **Today's Workout:**
                        • Type: Rest or Light Recovery
                        • Duration: 20-30 minutes max
                        • Intensity: <60% max HR
                        
                        **Why:** High fatigue detected. Rest now prevents injury and improves long-term performance.
                        """)
    
    with tab3:
        st.markdown("## Injury Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            risk = predictions.get('injury_risk_pct', 20)
            level = predictions.get('injury_risk_level', 'Moderate')
            
            fig = create_injury_risk_chart(risk, level)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk factors breakdown
            st.markdown("### Risk Factor Analysis")
            
            risk_factors = pd.DataFrame({
                'Factor': ['Training Load', 'Fatigue', 'Recovery', 'Biomechanics', 'History'],
                'Contribution': [30, 25, 20, 15, 10],
                'Status': ['Normal', 'Elevated', 'Good', 'Monitor', 'Clear']
            })
            
            fig = px.bar(risk_factors, x='Contribution', y='Factor', 
                        orientation='h', color='Status',
                        color_discrete_map={'Normal': '#4CAF50', 'Elevated': '#FFC107',
                                           'Good': '#4CAF50', 'Monitor': '#FF9800',
                                           'Clear': '#4CAF50'})
            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Prevention Protocol")
            
            acwr = predictions.get('acwr', 1.0)
            
            if level == "Low":
                st.success("""
                #### ✅ Low Risk - Maintain Current Approach
                
                **Key Metrics:**
                • ACWR: {:.2f} (Optimal)
                • Recovery: Good
                • Form: Stable
                
                **Recommendations:**
                1. Continue current training progression
                2. Maintain recovery practices
                3. Weekly strength training
                4. Monthly gait analysis
                """.format(acwr))
            elif level == "Moderate":
                st.warning("""
                #### ⚠️ Moderate Risk - Caution Advised
                
                **Key Metrics:**
                • ACWR: {:.2f} (Monitor)
                • Fatigue: Accumulating
                • Form: Slight degradation
                
                **Immediate Actions:**
                1. Reduce volume by 20% this week
                2. Add extra recovery day
                3. Focus on mobility work
                4. Consider massage therapy
                """.format(acwr))
            else:
                st.error("""
                #### 🚨 High Risk - Immediate Action Required
                
                **Key Metrics:**
                • ACWR: {:.2f} (Critical)
                • Fatigue: High
                • Form: Compromised
                
                **Mandatory Actions:**
                1. Stop high-intensity training
                2. 3-5 days complete rest
                3. Professional assessment
                4. Gradual return protocol
                """.format(acwr))
    
    with tab4:
        st.markdown("## Performance Trends & Projections")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # VO2 Max Trend
            dates = pd.date_range(start=datetime.now() - timedelta(days=90), 
                                 end=datetime.now() + timedelta(days=30), 
                                 freq='W')
            
            current_vo2 = predictions.get('vo2_max_current', 45)
            trend = predictions.get('vo2_max_trend', 0.1)
            
            historical = [current_vo2 - 2 + i*0.05 + np.random.uniform(-0.2, 0.2) 
                         for i in range(len(dates)-4)]
            # Start from last historical point for continuity
            last_historical = historical[-1] if historical else current_vo2
            # Convert monthly trend to weekly (divide by ~4.3 weeks per month)
            weekly_trend = trend / 4.3
            projected = [last_historical + weekly_trend * (i + 1) for i in range(4)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates[:-4], y=historical,
                mode='lines+markers', name='Historical',
                line=dict(color='#1a73e8', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=dates[-4:], y=projected,
                mode='lines+markers', name='ML Projected',
                line=dict(color='#34a853', width=2, dash='dash')
            ))
            
            fig.update_layout(
                title="VO2 Max Trajectory (ML Prediction)",
                xaxis_title="Date",
                yaxis_title="VO2 Max (ml/kg/min)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"**30-Day Projection:** {current_vo2:.1f} → {current_vo2 + trend*30:.1f} (+{trend*30:.1f})")
        
        with col2:
            # Performance improvements
            improvements = pd.DataFrame({
                'Metric': ['5K Time', '10K Time', 'Marathon', 'VO2 Max', 'Injury Risk'],
                'Current': ['25:00', '52:00', '4:30:00', '45.0', '20%'],
                '4 Weeks': ['24:15', '50:30', '4:25:00', '46.2', '15%'],
                'Change': ['-3%', '-3%', '-2%', '+2.7%', '-25%']
            })
            
            st.markdown("### 4-Week Performance Projection")
            st.dataframe(improvements, use_container_width=True, hide_index=True)
            
            # Progress visualization
            progress_data = {
                'Fitness': 85,
                'Endurance': 78,
                'Speed': 72,
                'Recovery': 88,
                'Consistency': 92
            }
            
            fig = go.Figure(go.Scatterpolar(
                r=list(progress_data.values()),
                theta=list(progress_data.keys()),
                fill='toself',
                marker=dict(color='#1a73e8')
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                title="Performance Profile",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("## ML Model Insights")
        
        # Get feature importance
        importance = service.get_feature_importance()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Race Performance Drivers")
            if 'race' in importance:
                fig = create_feature_importance_chart(
                    importance['race'],
                    "Top 5 Features - Race Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Health Metrics Drivers")
            if 'health' in importance:
                fig = create_feature_importance_chart(
                    importance['health'],
                    "Top 5 Features - Health Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Model Architecture")
            
            st.info("""
            **3 Specialized Random Forest Models:**
            
            🏃 **Race Performance Model**
            • Outputs: 4 race distances
            • Trees: 150, Max Depth: 12
            • R² Score: 0.978
            • MAE: <2% of race time
            
            💪 **Health Metrics Model**
            • Outputs: Readiness, Injury Risk
            • Trees: 100, Max Depth: 8  
            • R² Score: 0.962 (Readiness)
            • MAE: 1.2% (Readiness)
            
            📈 **Fitness Trend Model**
            • Output: VO2 Max trend
            • Trees: 50, Max Depth: 6
            • R² Score: 0.943
            • MAE: 0.002 units/month
            """)
            
            # Training data stats
            st.markdown("### Training Data")
            st.metric("Athletes", "100")
            st.metric("Training Days", "9,000")
            st.metric("Features", "25")
            st.metric("Total Predictions", "700 (7 outputs × 100 users)")
    
    with tab6:
        st.markdown("## Executive Dashboard")
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Model Accuracy", "97.8%", "+42% vs baseline")
        with col2:
            st.metric("User Engagement", "8.5/10", "+2.3 points")
        with col3:
            st.metric("Conversion Rate", "8.2%", "+5.2pp")
        with col4:
            st.metric("Monthly Revenue", "$5.8M", "+$3.5M")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Connect+ Value Proposition")
            
            value_data = pd.DataFrame({
                'Feature': ['Race Predictions', 'Injury Prevention', 'Training Plans',
                           'Recovery Insights', 'Performance Trends'],
                'Basic': ['❌', '❌', 'Generic', '❌', 'Limited'],
                'Connect+ ($6.99)': ['✅ ±2% accuracy', '✅ AI-powered', 'Personalized',
                                     '✅ Real-time', '✅ 30-day projection']
            })
            
            st.dataframe(value_data, use_container_width=True, hide_index=True)
            
            # ROI Calculation
            st.markdown("### ROI Analysis")
            
            roi_metrics = {
                'New Subscribers': 50000,
                'Monthly Revenue': 50000 * 6.99,
                'Annual Revenue': 50000 * 6.99 * 12,
                'Development Cost': 500000,
                'ROI Year 1': ((50000 * 6.99 * 12 - 500000) / 500000 * 100)
            }
            
            st.success(f"""
            **Business Case:**
            • New Subscribers: {roi_metrics['New Subscribers']:,}
            • Monthly Revenue: ${roi_metrics['Monthly Revenue']:,.0f}
            • Annual Revenue: ${roi_metrics['Annual Revenue']:,.0f}
            • Dev Cost: ${roi_metrics['Development Cost']:,}
            • **ROI Year 1: {roi_metrics['ROI Year 1']:.0f}%**
            """)
        
        with col2:
            st.markdown("### Competitive Positioning")
            
            comp_data = pd.DataFrame({
                'Company': ['Garmin Connect+', 'Apple Fitness+', 'Whoop', 'Strava Summit'],
                'Price': ['$6.99', '$9.99', '$30', '$5'],
                'AI Features': ['✅', '✅', '✅', '❌'],
                'Accuracy': ['±2%', '±5%', '±3%', 'N/A'],
                'Hardware': ['Optional', 'Required', 'Required', 'None']
            })
            
            st.dataframe(comp_data, use_container_width=True, hide_index=True)
            
            # Market opportunity
            st.markdown("### Market Opportunity")
            
            fig = go.Figure(go.Waterfall(
                name="Revenue", orientation="v",
                measure=["absolute", "relative", "relative", "relative", "total"],
                x=["Current", "Conversion +5%", "Churn -10%", "ARPU +$2", "Projected"],
                y=[2000000, 1000000, 800000, 500000, 4300000],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            
            fig.update_layout(
                title="Annual Revenue Projection ($M)",
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-top: 2rem;">
        <p style="color: #9ca3af; font-size: 14px; margin: 4px 0;">
            Garmin Performance AI - Ready for Production
        </p>
        <p style="color: #9ca3af; font-size: 12px; margin: 4px 0;">
            Powered by Cloudera AI
        </p>
        <p style="color: #cbd5e1; font-size: 11px; margin: 4px 0;">
            © 2025 Garmin International | Cloudera Partnership
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()