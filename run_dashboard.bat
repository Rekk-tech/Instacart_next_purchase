@echo off
echo Starting Instacart Analytics Dashboard...
echo =====================================

REM Check if virtual environment exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install dependencies
echo Installing dashboard dependencies...
pip install streamlit plotly altair streamlit-option-menu streamlit-aggrid

REM Start Streamlit dashboard
echo.
echo Starting dashboard at http://localhost:8501
echo Press Ctrl+C to stop the dashboard
echo.

streamlit run reports\dashboards\streamlit_app.py --server.port 8501 --server.address localhost

pause