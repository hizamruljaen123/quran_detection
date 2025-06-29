@echo off
echo.
echo =====================================================
echo 🧪 MEMORY FIX TEST SCRIPT
echo =====================================================
echo.

echo 🔍 Checking if Flask app is running...
curl -s http://127.0.0.1:5000 >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Flask app is not running!
    echo.
    echo Please start your Flask app first:
    echo    python app.py
    echo.
    echo Or run both in separate terminals:
    echo    Terminal 1: python app.py
    echo    Terminal 2: test_memory_fix.bat
    echo.
    pause
    exit /b 1
)

echo ✅ Flask app is running
echo.

echo 🧪 Running memory fix tests...
python test_memory_fix.py

echo.
echo =====================================================
echo 🏁 TEST COMPLETED
echo =====================================================

pause
