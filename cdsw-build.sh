#!/bin/bash
###############################
# CDSW Build Script for Garmin Performance AI
# This script prepares the environment for running the application
###############################

echo "================================================"
echo "Starting Garmin Performance AI Setup"
echo "================================================"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating required directories..."
mkdir -p data/sample_data
mkdir -p models/saved_models
mkdir -p .cache/llm_responses
mkdir -p demo/assets

# Check if data exists, if not generate it
if [ ! -f "data/sample_data/user_profiles.parquet" ]; then
    echo "Generating synthetic training data..."
    python data/synthetic_generator.py --users 100 --days 90 --output data/sample_data
else
    echo "Training data already exists."
fi

# Check if models exist, if not train them
if [ ! -f "models/saved_models/rf_race_model.pkl" ]; then
    echo "Training Random Forest models..."
    python models/train_rf_separate.py
else
    echo "Models already trained."
fi

# Verify OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY not set. LLM features will not work."
    echo "Please set it in your CML environment variables."
else
    echo "OpenAI API key detected."
fi

# Set streamlit config for CML
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
serverAddress = "0.0.0.0"
serverPort = 8501
EOF

echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. Ensure OPENAI_API_KEY is set in environment variables"
echo "2. Run 'streamlit run demo/streamlit_app_final.py' to start the app"
echo "3. Or use CML Applications to deploy the web interface"