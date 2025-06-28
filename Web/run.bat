@echo off
echo Starting Quran Verse Detection Web Application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH!
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "app.py" (
    echo Error: app.py not found!
    echo Please run this script from the Web directory.
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
python -c "import flask, numpy, librosa, tensorflow, sklearn, mysql.connector" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Create necessary directories
if not exist "static\uploads" mkdir static\uploads
if not exist "models" mkdir models

echo.
echo Starting Flask application...
echo Application will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
