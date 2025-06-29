@echo off
echo.
echo 🔍 QURAN DETECTION - CRASH DIAGNOSIS TOOL
echo ==========================================
echo.

echo 📋 Running comprehensive system diagnosis...
python debug_system.py

echo.
echo 🚨 Running post-prediction crash analysis...
python crash_analyzer.py

echo.
echo ✅ Diagnosis completed!
echo 📋 Check the recommendations above and apply the suggested fixes.
echo.
pause
