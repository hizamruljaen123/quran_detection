@echo off
echo.
echo 🛡️ QURAN DETECTION - ANTI-CRASH VERSION
echo ========================================
echo.

echo 📋 Starting Flask application with crash prevention...
echo.

echo 🔧 Configuration:
echo    - Error handling: ENABLED
echo    - Memory monitoring: ENABLED
echo    - Auto cleanup: ENABLED
echo    - Emergency recovery: ENABLED
echo.

echo 🌐 Server will start on: http://127.0.0.1:5000
echo 📊 Error monitor: http://127.0.0.1:5000/admin/errors
echo 💾 Memory status: http://127.0.0.1:5000/admin/memory
echo.

echo 🚨 IMPORTANT: Application will NOT crash even if errors occur!
echo    - All errors are caught and logged
echo    - Memory is automatically cleaned up
echo    - Flask process stays alive
echo.

echo ⏱️ Starting in 3 seconds...
timeout /t 3 /nobreak > nul

echo.
echo 🚀 STARTING FLASK APPLICATION...
echo ================================
python app.py

echo.
echo ⚠️ Flask application stopped.
echo 📋 Check error logs if unexpected shutdown occurred.
echo.
pause
