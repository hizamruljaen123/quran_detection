@echo off
echo ========================================
echo    Quran Verse Detection Web App
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH!
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo ✅ Python found

REM Check if we're in the right directory
if not exist "app.py" (
    echo ❌ app.py not found!
    echo Please run this script from the Web directory.
    pause
    exit /b 1
)

echo ✅ App files found

REM Run system diagnosis
echo.
echo 🔍 Running system diagnosis...
echo.
python debug_system.py
echo.

REM Ask user if they want to continue
set /p continue="Continue with app startup? (y/n): "
if /i not "%continue%"=="y" (
    echo App startup cancelled
    pause
    exit /b 0
)

REM Check if requirements are installed
echo.
echo 📦 Checking dependencies...
python -c "import flask, numpy, librosa, tensorflow, sklearn, mysql.connector" >nul 2>&1
if errorlevel 1 (
    echo Installing/updating dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies!
        pause
        exit /b 1
    )
    echo ✅ Dependencies installed
) else (
    echo ✅ Dependencies OK
)

REM Create necessary directories
echo.
echo 📁 Setting up directories...
if not exist "static\uploads" (
    mkdir static\uploads
    echo ✅ Created uploads directory
)
if not exist "models" (
    mkdir models
    echo ✅ Created models directory
)

echo.
echo 🚀 Starting Flask application...
echo 📖 Application will be available at: http://localhost:5000
echo 🛑 Press Ctrl+C to stop the server
echo.

python app.py

echo.
echo 🛑 Server stopped
pause
