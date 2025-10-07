# Garmin Performance AI

A proof-of-concept for predicting running performance using machine learning. Built for Garmin Connect+ to help runners train smarter.

## What it does

- Predicts race finish times (5K, 10K, Half Marathon, Marathon)
- Provides training readiness scores
- Assesses injury risk
- Suggests optimal pacing strategies
- Projects fitness trends

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the demo
streamlit run demo/streamlit_app_final.py
```

Then open http://localhost:8501 in your browser.

## Tech Stack

- Python with Streamlit for the web interface
- Random Forest models for predictions
- OpenAI GPT-4 for natural language explanations
- Synthetic training data for 100 sample athletes

## Project Structure

```
demo/                 # Streamlit web app
models/               # ML models and training code
data/                 # Data generation and processing
explainer/            # AI coach and explanations
```

## Demo Users

Try these personas in the app:
- Elite Emma - competitive athlete
- Recreational Rachel - weekend warrior  
- Beginner Ben - just starting out

## Author

Neelabh Pant