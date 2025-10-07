# Garmin Performance AI - POC Project

## Project Overview
Building a Performance Prediction POC for Garmin's Fitness division to demonstrate Cloudera Machine Learning (CML) capabilities. This is a demo to show how their existing 1200-node CDP infrastructure can power AI features for their Connect+ premium subscription ($6.99/month).

## Critical Context
- **My Role**: Director, Global AI Industry Solutions at Cloudera (Retail focus)
- **Meeting Type**: Follow-up POC after initial Garmin meeting
- **Client Status**: Existing Cloudera customer with massive on-prem investment
- **Deadline**: 2-week POC to show value
- **Success Criteria**: Demonstrate clear ROI for Connect+ subscribers using AI

## Business Context
- **Client**: Garmin - $6B+ revenue, 300M devices sold
- **Division**: Fitness (28% of revenue, $1.77B annually)  
- **Infrastructure**: 1200 on-prem CDP nodes in Kansas City (must stay on-premises)
- **Product**: Garmin Connect+ premium subscription ($6.99/month)
- **Competition**: Apple Watch AI, Whoop ($30/month), Strava
- **Key Initiative**: "Active Intelligence" - AI-powered insights for fitness users
- **Goal**: Increase Connect+ conversion from 3% to 8%, reduce churn from 15% to <5%

## Technical Architecture
Building a Neural Network-based solution with LLM explanations:
1. **Synthetic Data Generation**: 100 users × 90 days of training data
2. **Neural Network Model**: Multi-output predictions (single model, multiple outputs)
3. **LLM Explainer**: Natural language explanations using GPT-4/Claude
4. **Demo Interface**: Streamlit app showcasing real-time predictions
5. **CDP Integration Story**: How this deploys on their existing infrastructure

## Core Features to Implement

### 1. Race Finish Time Prediction
- Predict 5K, 10K, Half Marathon, Marathon times
- ±5% accuracy target
- Based on training history, VO2 max, recent performance

### 2. Training Readiness Score (0-100)
- Combines acute:chronic workload ratio, HRV trends, sleep, recovery
- Traffic light system: Green (>80), Yellow (60-80), Red (<60)
- Updates daily based on new training data

### 3. Injury Risk Assessment  
- Three levels: Low (<20%), Moderate (20-50%), High (>50%)
- Based on training load spikes, biomechanical imbalances, fatigue
- Includes specific concerns (e.g., "High mileage increase detected")

### 4. Optimal Pacing Strategy
- 10 splits for any race distance
- Accounts for user's fade tendency
- Negative split vs even pacing recommendation

### 5. VO2 Max Trajectory
- 30-day projection
- Shows if training is improving fitness
- Compared to age/gender norms

## Project Structure
```
garmin-performance-ai/
├── claude.md                    # This file - project context
├── requirements.txt             # Python dependencies  
├── .env                        # API keys (OpenAI)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
│
├── data/
│   ├── __init__.py
│   ├── synthetic_generator.py  # Generate realistic training data
│   ├── feature_engineering.py  # Feature preparation pipeline
│   ├── data_schemas.py        # Pydantic schemas for validation
│   └── sample_data/            # Generated CSV/parquet files
│       └── .gitkeep
│
├── models/
│   ├── __init__.py
│   ├── performance_nn.py      # Multi-output neural network
│   ├── training.py            # Training pipeline with MLflow
│   ├── inference.py           # Prediction service
│   └── saved_models/          # Trained model files (.h5, .pkl)
│       └── .gitkeep
│
├── explainer/
│   ├── __init__.py
│   ├── llm_explainer.py       # GPT-4o
│   ├── prompts.py             # Prompt templates
│   └── feature_importance.py  # SHAP/gradient-based importance
│
├── demo/
│   ├── __init__.py
│   ├── streamlit_app.py       # Main demo interface
│   ├── visualizations.py      # Plotly charts and graphs
│   ├── sample_personas.py     # Pre-built user profiles
│   └── assets/                # Images, logos, CSS
│       ├── garmin_logo.png
│       └── custom.css
│
├── cdp_integration/
│   ├── __init__.py
│   ├── cml_simulation.py      # Simulate CML deployment
│   ├── data_pipeline.py       # Simulate CDP data flow
│   └── deployment_guide.md    # How to deploy on real CDP
│
├── tests/
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_model.py
│   └── test_predictions.py
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_model_development.ipynb
    └── 03_demo_testing.ipynb
```

## Technical Stack
- **Python**: Python 3.13.3
- **ML Framework**: TensorFlow/Keras for neural network
- **Data Processing**: Pandas, NumPy, Polars (for speed)
- **Feature Store**: Local parquet files (simulating Iceberg tables)
- **Visualization**: Plotly, Streamlit
- **LLM**: OpenAI GPT-4o for explanations
- **MLOps**: MLflow for experiment tracking (simulating CML)
- **Testing**: Pytest, pytest-cov
- **Deployment Target**: Cloudera Machine Learning (CML)

## Data Schemas

### User Profile Schema
```python
{
    "user_id": str,              # "USER_0001"
    "age": int,                  # 18-70
    "gender": str,               # "M" or "F"
    "height_cm": float,          # 150-200
    "weight_kg": float,          # 45-120
    "vo2_max_baseline": float,   # 30-70 ml/kg/min
    "weekly_miles_baseline": float, # 5-80 miles
    "injury_prone": bool,        # History of injuries
    "years_running": float,      # Experience level
    "persona_type": str          # "elite", "competitive", "recreational", "beginner"
}
```

### Training Data Schema
```python
{
    "user_id": str,
    "date": datetime,
    "workout_type": str,         # "easy", "tempo", "intervals", "long", "recovery", "rest"
    "distance_km": float,        
    "duration_min": float,
    "avg_hr": int,              # Average heart rate
    "max_hr": int,              # Maximum heart rate
    "avg_pace_min_per_km": float,
    "elevation_gain_m": float,
    "hrv_morning": float,        # Heart rate variability
    "sleep_hours": float,
    "stress_score": float,       # 0-10
    "temperature_c": float,      # Weather
    "humidity_pct": float,
    "fatigue_level": float,      # 0-10
    "current_vo2_max": float     # Estimated current VO2max
}
```

### Prediction Output Schema
```python
{
    "race_time_seconds": float,
    "race_time_formatted": str,   # "3:45:23"
    "readiness_score": int,       # 0-100
    "injury_risk_pct": float,     # 0-100
    "injury_risk_level": str,     # "Low", "Moderate", "High"
    "pace_splits": List[float],   # 10 segments
    "vo2_max_trend": float,       # Change per month
    "explanation": str,           # LLM generated text
    "confidence_interval": tuple  # (lower_bound, upper_bound)
}
```

## Key Algorithms/Formulas

### Acute:Chronic Workload Ratio (ACWR)
```python
# Injury risk predictor
acute_load = sum(last_7_days_training)
chronic_load = sum(last_28_days_training) / 4
acwr = acute_load / chronic_load

# Risk zones:
# <0.8: Undertraining (detraining risk)
# 0.8-1.3: Optimal (sweet spot)
# 1.3-1.5: Moderate injury risk
# >1.5: High injury risk
```

### Race Time Prediction Base
```python
# VO2max-based prediction (Jack Daniels' formula variant)
velocity = (vo2max - 3.5) / 0.182
race_velocity = velocity * efficiency_factor * distance_factor

# Distance factors:
# 5K: 0.95
# 10K: 0.92  
# Half: 0.88
# Full: 0.83
```

### Training Effect
```python
# TRIMP (Training Impulse) calculation
trimp = duration * hr_reserve_fraction * gender_factor
training_effect = trimp * (1 - fatigue/10) * adaptation_rate

# Fitness change
fitness_new = fitness_old + training_effect - decay_rate
```

### Readiness Score
```python
readiness = 100  # Start at perfect
readiness -= max(0, (acwr - 1.3) * 40)  # Penalize high load
readiness -= max(0, (0.8 - acwr) * 30)  # Penalize low load  
readiness -= max(0, (baseline_hrv - current_hrv) / baseline_hrv * 30)
readiness -= max(0, (7.5 - sleep_hours) * 5)
readiness = max(0, min(100, readiness))
```

## Development Phases

### Phase 1: Data Foundation
- [x] Create project structure
- [ ] Generate synthetic training data for 100 users
- [ ] Create diverse user personas
- [ ] Generate historical race results
- [ ] Validate data distributions

### Phase 2: Model Development
- [ ] Build multi-output neural network
- [ ] Implement feature engineering pipeline
- [ ] Train model on synthetic data
- [ ] Validate predictions against known formulas
- [ ] Save model artifacts

### Phase 3: Explainability
- [ ] Integrate LLM API (GPT-4o)
- [ ] Create explanation templates
- [ ] Add feature importance calculation
- [ ] Generate personalized insights

### Phase 4: Demo Interface
- [ ] Build Streamlit dashboard
- [ ] Create interactive visualizations
- [ ] Add sample personas
- [ ] Polish UI/UX
- [ ] Add Garmin branding

### Phase 5: CDP Integration Story
- [ ] Document CML deployment process
- [ ] Show MLflow integration
- [ ] Create scaling projections
- [ ] Build executive presentation

## Success Metrics
- **Model Performance**:
  - Race time prediction: ±5% accuracy
  - Injury risk: >80% precision for high risk
  - Readiness score: Correlates with performance (r>0.7)
  
- **Technical Performance**:
  - Inference latency: <100ms
  - Model size: <10MB (edge deployable)
  - Training time: <1 hour on GPU
  
- **Demo Impact**:
  - Clear value proposition for Connect+
  - Differentiation from Apple/Whoop
  - Executive-ready presentation
  - Shows CDP/CML capabilities

## CDP/Cloudera Integration Points

### How This Maps to CDP Components
1. **Data Flow (NiFi/Kafka)**: Real-time sensor data ingestion
2. **Data Engineering (Spark)**: Feature computation at scale
3. **Data Warehouse (Impala)**: Historical analysis queries
4. **Machine Learning (CML)**: Model training and serving
5. **Operational DB (HBase)**: User predictions cache
6. **Data Catalog (Atlas)**: Feature lineage tracking

### Deployment on CML
```python
# Show this in demo
from cmlapi import CMLServiceClient

client = CMLServiceClient()
model = client.deploy_model(
    model_path='models/saved_models/performance_model.h5',
    serving_framework='tensorflow',
    cpu_cores=4,
    memory_gb=8, 
    replicas=3,
    autoscaling=True,
    min_replicas=1,
    max_replicas=10
)
```

### Scaling Projections
- 100 pilot users → 10K beta users → 100M production users
- 1 model → User segment models → Personalized models
- Batch daily → Streaming updates → Real-time edge scoring

## Important Notes & Constraints
1. **On-premises only**: Garmin won't use cloud services
2. **Data sovereignty**: User fitness data cannot leave Kansas City
3. **Existing investment**: Must leverage their 1200 CDP nodes
4. **Premium positioning**: Features must justify $6.99/month
5. **Competitive pressure**: Apple/WHOOP investing billions in health AI
6. **User trust**: Explanations must be clear and accurate

## Demo Talking Points
1. "This runs on your existing CDP infrastructure"
2. "No new hardware or cloud services required"
3. "Processes billions of sensor readings daily"
4. "Scales from 100 to 100M users"
5. "Deploys in weeks, not months"
6. "Bank-grade security for health data"
7. "Differentiates Connect+ from free tier"
8. "Matches Apple/Whoop capabilities at lower cost"

## Common Development Commands
```bash
# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Generate synthetic data
python data/synthetic_generator.py --users 100 --days 90 --output data/sample_data/

# Train model
python models/training.py --data data/sample_data/ --epochs 50 --batch-size 32

# Run tests
pytest tests/ -v --cov=.

# Start demo
streamlit run demo/streamlit_app.py --server.port 8501

# MLflow UI (simulating CML)
mlflow ui --port 5000

# Format code
black . --line-length 100
isort . --profile black

# Type checking
mypy . --ignore-missing-imports
```

## Environment Variables (.env)
```bash
# LLM APIs
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Demo Configuration
DEMO_MODE=true
USE_CACHED_PREDICTIONS=false
SHOW_TECHNICAL_DETAILS=true

# Model Settings
MODEL_VERSION=v1.0
CONFIDENCE_LEVEL=0.90

# Garmin Branding
PRIMARY_COLOR=#1a73e8
SECONDARY_COLOR=#f5f5f5
```

## Questions/Decisions to Address
1. **LLM Choice**: Use GPT-4o for explanations? 
2. **Data Volume**: 100 users sufficient for demo? (Yes, but mention scale potential)
3. **Real-time vs Batch**: Show both? (Yes, batch for POC, mention streaming capability)
4. **Edge Deployment**: Include watch deployment story? (Yes, as future phase)
5. **Competitive Features**: Which Apple/Whoop features to highlight as matched?
6. **Pricing Model**: Show ROI calculation for Connect+?
7. **Performance Benchmarks**: Include latency/throughput metrics?
8. **User Personas**: How many to pre-build? (4-5 covering different segments)


## Contact & Resources
- **Project Lead**: Neelabh Pant - Director, Global AI Industry Solutions
- **Cloudera CML Docs**: https://docs.cloudera.com/machine-learning/
- **Garmin Connect API**: Internal documentation needed
- **Competition Analysis**: See competitive_analysis.md

---
*This POC demonstrates how Cloudera's integrated data platform transforms fragmented fitness data into measurable AI outcomes, justifying Garmin's Connect+ premium subscription and competing with Apple/Whoop innovations.*