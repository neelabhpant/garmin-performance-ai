"""Visualization utilities for the Streamlit demo."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def create_readiness_gauge(score: int, status: str) -> go.Figure:
    """Create a gauge chart for readiness score."""
    
    colors = {
        'green': '#4CAF50',
        'yellow': '#FFC107',
        'red': '#F44336'
    }
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "Readiness Score"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': colors[status]},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 60], 'color': '#ffebee'},
                {'range': [60, 80], 'color': '#fff9c4'},
                {'range': [80, 100], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    
    return fig


def create_injury_risk_chart(risk: float, level: str) -> go.Figure:
    """Create injury risk visualization."""
    
    colors = {
        'Low': '#4CAF50',
        'Moderate': '#FFC107', 
        'High': '#F44336'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[risk],
        y=['Current Risk'],
        orientation='h',
        marker=dict(
            color=colors[level],
            line=dict(color='rgba(0,0,0,0.3)', width=2)
        ),
        text=f"{risk:.0f}%",
        textposition='inside',
        textfont=dict(size=20, color='white')
    ))
    
    fig.add_vline(x=20, line_width=2, line_dash="dash", line_color="gray",
                  annotation_text="Low", annotation_position="top")
    fig.add_vline(x=50, line_width=2, line_dash="dash", line_color="gray",
                  annotation_text="Moderate", annotation_position="top")
    
    fig.update_layout(
        title=f"Injury Risk Level: {level}",
        xaxis=dict(range=[0, 100], title="Risk Percentage"),
        yaxis=dict(showticklabels=False),
        height=200,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False
    )
    
    return fig


def create_vo2_gauge(vo2_value: float) -> go.Figure:
    """Create a small gauge chart for VO2 Max metric."""
    
    # Determine category based on VO2 Max value
    if vo2_value >= 55:
        color = '#4CAF50'  # Green - Excellent
        category = 'Excellent'
    elif vo2_value >= 45:
        color = '#2196F3'  # Blue - Good
        category = 'Good'
    elif vo2_value >= 35:
        color = '#FFC107'  # Yellow - Fair
        category = 'Fair'
    else:
        color = '#FF9800'  # Orange - Below Average
        category = 'Below Avg'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=vo2_value,
        title={'text': f"VO2 Max<br><span style='font-size:12px'>{category}</span>", 'font': {'size': 16}},
        delta={'reference': 45, 'relative': False},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [25, 70], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [25, 35], 'color': '#ffe0b2'},
                {'range': [35, 45], 'color': '#fff3e0'},
                {'range': [45, 55], 'color': '#e3f2fd'},
                {'range': [55, 70], 'color': '#e8f5e9'}
            ],
        }
    ))
    
    fig.update_layout(
        height=180, 
        margin=dict(l=30, r=30, t=90, b=30),
        font=dict(size=13)
    )
    
    return fig


def create_mileage_gauge(miles: float) -> go.Figure:
    """Create a small gauge chart for Weekly Mileage."""
    
    # Determine training phase based on mileage
    if miles >= 60:
        color = '#9C27B0'  # Purple - Peak
        phase = 'Peak Phase'
    elif miles >= 40:
        color = '#3F51B5'  # Indigo - Build
        phase = 'Build Phase'
    elif miles >= 20:
        color = '#00BCD4'  # Cyan - Base
        phase = 'Base Phase'
    else:
        color = '#607D8B'  # Blue Grey - Recovery
        phase = 'Recovery'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=miles,
        number={'suffix': " mi"},
        title={'text': f"Weekly Miles<br><span style='font-size:12px'>{phase}</span>", 'font': {'size': 16}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 80], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [0, 20], 'color': '#eceff1'},
                {'range': [20, 40], 'color': '#e0f7fa'},
                {'range': [40, 60], 'color': '#e8eaf6'},
                {'range': [60, 80], 'color': '#f3e5f5'}
            ],
        }
    ))
    
    fig.update_layout(
        height=180, 
        margin=dict(l=30, r=30, t=90, b=30),
        font=dict(size=13)
    )
    
    return fig


def create_pace_gauge(pace: float) -> go.Figure:
    """Create a small gauge chart for Average Pace."""
    
    # Convert pace to speed for better visualization (lower pace = higher speed)
    # Invert the scale: 4 min/km = fast, 8 min/km = slow
    speed_score = max(0, min(100, (8 - pace) / 4 * 100))
    
    # Determine zone based on pace
    if pace <= 4.5:
        color = '#F44336'  # Red - Race Pace
        zone = 'Race Pace'
    elif pace <= 5.5:
        color = '#FF9800'  # Orange - Threshold
        zone = 'Threshold'
    elif pace <= 6.5:
        color = '#4CAF50'  # Green - Tempo
        zone = 'Tempo'
    else:
        color = '#2196F3'  # Blue - Easy
        zone = 'Easy Pace'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pace,
        number={'suffix': " min/km"},
        title={'text': f"Avg Pace<br><span style='font-size:12px'>{zone}</span>", 'font': {'size': 16}},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [3.5, 8], 'tickwidth': 1, 'dtick': 0.5},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e0e0e0",
            'steps': [
                {'range': [3.5, 4.5], 'color': '#ffebee'},
                {'range': [4.5, 5.5], 'color': '#fff3e0'},
                {'range': [5.5, 6.5], 'color': '#e8f5e9'},
                {'range': [6.5, 8], 'color': '#e3f2fd'}
            ],
        }
    ))
    
    fig.update_layout(
        height=180, 
        margin=dict(l=30, r=30, t=90, b=30),
        font=dict(size=13)
    )
    
    return fig


def create_vo2_trend(historical_data: list, projected_data: list, dates: list) -> go.Figure:
    """Create VO2 max trend visualization."""
    
    fig = go.Figure()
    
    historical_dates = dates[:len(historical_data)]
    projected_dates = dates[len(historical_data):]
    
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_data,
        mode='lines+markers',
        name='Historical',
        line=dict(color='#1a73e8', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=projected_dates,
        y=projected_data,
        mode='lines+markers',
        name='Projected',
        line=dict(color='#34a853', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=projected_dates,
        y=[projected_data[0] - 1, projected_data[-1] - 1],
        fill='tonexty',
        mode='none',
        name='Confidence Band',
        fillcolor='rgba(52, 168, 83, 0.2)'
    ))
    
    fig.add_trace(go.Scatter(
        x=projected_dates,
        y=[projected_data[0] + 1, projected_data[-1] + 1],
        fill='tonexty',
        mode='none',
        showlegend=False,
        fillcolor='rgba(52, 168, 83, 0.2)'
    ))
    
    fig.update_layout(
        title="VO2 Max Trend & 30-Day Projection",
        xaxis_title="Date",
        yaxis_title="VO2 Max (ml/kg/min)",
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def create_pace_splits_chart(splits: list, distance_name: str) -> go.Figure:
    """Create pace splits visualization for races."""
    
    split_labels = [f"Split {i+1}" for i in range(len(splits))]
    pace_per_km = [s/60/4.2195 for s in splits]
    
    colors = ['#1a73e8' if i < len(splits)//2 else '#34a853' for i in range(len(splits))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=split_labels,
        y=pace_per_km,
        marker_color=colors,
        text=[f"{p:.2f}" for p in pace_per_km],
        textposition='outside'
    ))
    
    avg_pace = sum(pace_per_km) / len(pace_per_km)
    fig.add_hline(y=avg_pace, line_width=2, line_dash="dash", line_color="red",
                  annotation_text=f"Avg: {avg_pace:.2f} min/km")
    
    fig.update_layout(
        title=f"{distance_name} Pacing Strategy",
        xaxis_title="Split",
        yaxis_title="Pace (min/km)",
        height=400,
        showlegend=False
    )
    
    return fig


def create_training_load_chart(acute_load: float, chronic_load: float, acwr: float) -> go.Figure:
    """Create training load balance visualization."""
    
    fig = go.Figure()
    
    categories = ['Acute Load\n(7 days)', 'Chronic Load\n(28 days)']
    values = [acute_load, chronic_load]
    
    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker_color=['#FF6B6B', '#4ECDC4'],
        text=[f"{v:.0f}" for v in values],
        textposition='outside'
    ))
    
    optimal_color = '#4CAF50' if 0.8 <= acwr <= 1.3 else '#FFC107' if 0.5 <= acwr <= 1.5 else '#F44336'
    
    fig.add_annotation(
        x=0.5, y=max(values) * 1.1,
        text=f"ACWR: {acwr:.2f}",
        showarrow=False,
        font=dict(size=16, color=optimal_color),
        bgcolor=optimal_color,
        opacity=0.2,
        bordercolor=optimal_color,
        borderwidth=2,
        borderpad=4
    )
    
    fig.update_layout(
        title="Training Load Balance",
        yaxis_title="Arbitrary Units",
        height=300,
        showlegend=False,
        margin=dict(t=80)
    )
    
    return fig