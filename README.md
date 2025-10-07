# Garmin Performance AI - POC Project

## Overview
A Neural Network-based Performance Prediction system for Garmin Connect+ premium subscription, demonstrating Cloudera Machine Learning (CML) capabilities on Garmin's existing 1200-node CDP infrastructure.

## Business Value
- **Target**: Increase Connect+ conversion from 3% to 8%, reduce churn from 15% to <5%
- **ROI**: Justifies $6.99/month premium subscription
- **Competition**: Matches Apple Watch AI and Whoop ($30/month) capabilities

## Features
- **Race Finish Time Prediction**: 5K, 10K, Half Marathon, Marathon (±5% accuracy)
- **Training Readiness Score**: Daily 0-100 score with traffic light system
- **Injury Risk Assessment**: Low/Moderate/High risk levels with specific concerns
- **Optimal Pacing Strategy**: 10 splits for any race distance
- **VO2 Max Trajectory**: 30-day fitness projection

## Quick Start

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your OpenAI API key to .env
```

### Generate Synthetic Data
```bash
python data/synthetic_generator.py --users 100 --days 90 --output data/sample_data/
```

### Train Model
```bash
python models/training.py --data data/sample_data/ --epochs 50 --batch-size 32
```

### Run Demo
```bash
streamlit run demo/streamlit_app.py --server.port 8501
```

## Project Structure
```
├── data/                   # Data generation and processing
├── models/                 # Neural network architecture
├── explainer/              # LLM-based explanations
├── demo/                   # Streamlit demo interface
├── cdp_integration/        # Cloudera deployment guides
├── tests/                  # Test suite
└── notebooks/              # Development notebooks
```

## Technical Stack
- **Python 3.13.3** with TensorFlow/Keras
- **Streamlit** for interactive demo
- **OpenAI GPT-4o** for explanations
- **MLflow** for experiment tracking
- **Deployment**: Cloudera Machine Learning (CML)

## CDP Integration
Designed to run on Garmin's existing 1200-node CDP infrastructure:
- **Data Flow**: NiFi/Kafka for real-time sensor ingestion
- **Data Engineering**: Spark for feature computation
- **Machine Learning**: CML for model training/serving
- **Data Warehouse**: Impala for historical analysis

## Performance Targets
- Model accuracy: ±5% for race predictions
- Inference latency: <100ms
- Model size: <10MB (edge deployable)
- Scales: 100 → 10K → 100M users

## Development Team
- **Project Lead**: Neelabh Pant - Director, Global AI Industry Solutions, Cloudera
- **Focus**: Retail & Fitness Industry Solutions
- **Timeline**: 2-week POC

## License
Proprietary - Cloudera & Garmin Confidential