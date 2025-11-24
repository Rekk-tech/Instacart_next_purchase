#!/bin/bash

echo "Starting Instacart Analytics Dashboard..."
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dashboard dependencies..."
pip install streamlit plotly altair streamlit-option-menu streamlit-aggrid

# Start Streamlit dashboard
echo ""
echo "Starting dashboard at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run reports/dashboards/streamlit_app.py --server.port 8501 --server.address localhost