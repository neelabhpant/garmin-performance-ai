"""Modern visualization components inspired by Whoop and Fitbit."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Optional, List, Dict, Tuple


def create_circular_progress(value: float, max_value: float = 100, 
                            title: str = "", color: str = "#667eea",
                            size: str = "medium") -> go.Figure:
    """Create a circular progress indicator like Whoop's recovery/strain rings.
    
    Args:
        value: Current value
        max_value: Maximum value for percentage calculation
        title: Title to display
        color: Primary color for the ring
        size: Size of the chart ('small', 'medium', 'large')
    """
    sizes = {"small": 200, "medium": 300, "large": 400}
    chart_size = sizes.get(size, 300)
    
    percentage = min(100, (value / max_value) * 100)
    
    # Create the ring
    fig = go.Figure()
    
    # Background ring (gray)
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        text=[f"<b>{value:.0f}</b>"],
        textfont=dict(size=48, color=color),
        mode='text',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add percentage ring using pie chart
    colors = [color, 'rgba(0,0,0,0.05)']
    
    fig.add_trace(go.Pie(
        values=[percentage, 100-percentage],
        hole=0.75,
        marker=dict(colors=colors, line=dict(width=0)),
        textinfo='none',
        hoverinfo='skip',
        sort=False,
        direction='clockwise',
        rotation=90
    ))
    
    # Add subtitle
    fig.add_annotation(
        text=title,
        x=0.5, y=0.3,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color="#6b7280")
    )
    
    # Add percentage text
    fig.add_annotation(
        text=f"{percentage:.0f}%",
        x=0.5, y=0.42,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="#9ca3af")
    )
    
    fig.update_layout(
        height=chart_size,
        width=chart_size,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


def create_mini_sparkline(data: List[float], color: str = "#667eea",
                         show_area: bool = True) -> go.Figure:
    """Create a mini sparkline chart for trend visualization.
    
    Args:
        data: List of values to plot
        color: Line color
        show_area: Whether to show area under the line
    """
    fig = go.Figure()
    
    x = list(range(len(data)))
    
    if show_area:
        fig.add_trace(go.Scatter(
            x=x, y=data,
            mode='lines',
            line=dict(color=color, width=2),
            fill='tozeroy',
            fillcolor=f'{color}20',
            showlegend=False,
            hovertemplate='%{y:.1f}<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=data,
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False,
            hovertemplate='%{y:.1f}<extra></extra>'
        ))
    
    # Add dot for current value
    fig.add_trace(go.Scatter(
        x=[x[-1]], y=[data[-1]],
        mode='markers',
        marker=dict(color=color, size=6),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hovermode='x'
    )
    
    return fig


def create_gradient_bar(value: float, ranges: List[Tuple[float, float, str]], 
                       labels: Optional[List[str]] = None) -> go.Figure:
    """Create a gradient bar indicator like Whoop's strain scale.
    
    Args:
        value: Current value to indicate
        ranges: List of (min, max, color) tuples for each range
        labels: Optional labels for each range
    """
    fig = go.Figure()
    
    # Create gradient background
    for i, (range_min, range_max, color) in enumerate(ranges):
        fig.add_shape(
            type="rect",
            x0=range_min, x1=range_max,
            y0=0, y1=1,
            fillcolor=color,
            opacity=0.6,
            layer="below",
            line_width=0,
        )
        
        if labels and i < len(labels):
            fig.add_annotation(
                text=labels[i],
                x=(range_min + range_max) / 2,
                y=0.5,
                showarrow=False,
                font=dict(size=10, color="white")
            )
    
    # Add indicator line
    fig.add_shape(
        type="line",
        x0=value, x1=value,
        y0=-0.1, y1=1.1,
        line=dict(color="black", width=3)
    )
    
    # Add value annotation
    fig.add_annotation(
        text=f"{value:.1f}",
        x=value,
        y=1.3,
        showarrow=False,
        font=dict(size=14, color="black", weight=600)
    )
    
    fig.update_layout(
        height=80,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            range=[ranges[0][0], ranges[-1][1]],
            visible=False
        ),
        yaxis=dict(
            range=[0, 1],
            visible=False
        )
    )
    
    return fig


def create_sleep_chart(sleep_data: Dict) -> go.Figure:
    """Create a sleep quality chart like Fitbit's sleep insights.
    
    Args:
        sleep_data: Dictionary with sleep stages and durations
    """
    stages = ['Awake', 'REM', 'Light', 'Deep']
    colors = ['#ef4444', '#f59e0b', '#3b82f6', '#8b5cf6']
    
    # Sample data structure
    if not sleep_data:
        sleep_data = {
            'Awake': [0.5, 0.2, 0.1, 0.3, 0.2],
            'REM': [1.5, 0, 0.8, 0, 1.2],
            'Light': [2.0, 3.5, 2.8, 3.0, 2.5],
            'Deep': [1.0, 1.3, 0.8, 1.2, 0.6]
        }
    
    fig = go.Figure()
    
    for stage, color in zip(stages, colors):
        if stage in sleep_data:
            fig.add_trace(go.Bar(
                name=stage,
                x=list(range(len(sleep_data[stage]))),
                y=sleep_data[stage],
                marker_color=color,
                hovertemplate=f'{stage}: %{{y:.1f}} hrs<extra></extra>'
            ))
    
    fig.update_layout(
        barmode='stack',
        height=250,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="",
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            title="Hours",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


def create_heart_rate_zones(hr_data: List[float], zones: Dict[str, Tuple[int, int]]) -> go.Figure:
    """Create a heart rate zones chart like Garmin/Whoop.
    
    Args:
        hr_data: List of heart rate values
        zones: Dictionary of zone names and (min, max) HR values
    """
    default_zones = {
        'Recovery': (0, 120),
        'Easy': (120, 140),
        'Moderate': (140, 160),
        'Hard': (160, 180),
        'Max': (180, 200)
    }
    
    zones = zones or default_zones
    zone_colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    
    fig = go.Figure()
    
    # Add shaded zones
    for (zone_name, (zone_min, zone_max)), color in zip(zones.items(), zone_colors):
        fig.add_shape(
            type="rect",
            x0=0, x1=len(hr_data),
            y0=zone_min, y1=zone_max,
            fillcolor=color,
            opacity=0.1,
            layer="below",
            line_width=0
        )
        
        fig.add_annotation(
            text=zone_name,
            x=len(hr_data) * 0.98,
            y=(zone_min + zone_max) / 2,
            xanchor="right",
            showarrow=False,
            font=dict(size=10, color=color)
        )
    
    # Add heart rate line
    fig.add_trace(go.Scatter(
        x=list(range(len(hr_data))),
        y=hr_data,
        mode='lines',
        line=dict(color='#ef4444', width=2),
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)',
        name='Heart Rate',
        hovertemplate='HR: %{y} bpm<extra></extra>'
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=50, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Time",
            showgrid=False
        ),
        yaxis=dict(
            title="Heart Rate (bpm)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        showlegend=False,
        hovermode='x'
    )
    
    return fig


def create_training_load_chart(acute: List[float], chronic: List[float]) -> go.Figure:
    """Create a training load chart showing acute vs chronic workload.
    
    Args:
        acute: List of acute training load values
        chronic: List of chronic training load values
    """
    fig = go.Figure()
    
    x = list(range(len(acute)))
    
    # Add chronic load (background)
    fig.add_trace(go.Scatter(
        x=x, y=chronic,
        mode='lines',
        name='Chronic Load',
        line=dict(color='#9ca3af', width=2, dash='dash'),
        hovertemplate='Chronic: %{y:.1f}<extra></extra>'
    ))
    
    # Add acute load
    fig.add_trace(go.Scatter(
        x=x, y=acute,
        mode='lines',
        name='Acute Load',
        line=dict(color='#667eea', width=3),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        hovertemplate='Acute: %{y:.1f}<extra></extra>'
    ))
    
    # Add optimal range
    optimal_min = [c * 0.8 for c in chronic]
    optimal_max = [c * 1.3 for c in chronic]
    
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=optimal_max + optimal_min[::-1],
        fill='toself',
        fillcolor='rgba(16, 185, 129, 0.1)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip',
        name='Optimal Range'
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Days",
            showgrid=False
        ),
        yaxis=dict(
            title="Training Load (AU)",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.05)'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    return fig


def create_performance_radar(metrics: Dict[str, float]) -> go.Figure:
    """Create a radar chart for performance metrics like Fitbit's wellness score.
    
    Args:
        metrics: Dictionary of metric names and values (0-100)
    """
    fig = go.Figure()
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='#667eea', width=2),
        marker=dict(color='#667eea', size=8),
        hovertemplate='%{theta}: %{r:.0f}%<extra></extra>'
    ))
    
    # Add reference circle at 50%
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(categories),
        theta=categories,
        mode='lines',
        line=dict(color='rgba(0,0,0,0.1)', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.05)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        height=400,
        margin=dict(l=80, r=80, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig