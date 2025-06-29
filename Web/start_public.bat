REM Run Quran Detection App with Ngrok tunnel only

echo.
set /p ngrok_token=Enter your Ngrok auth token (or press Enter to skip): 
echo.

echo Starting app with Ngrok tunnel...
echo.

if "%ngrok_token%"=="" (
    python app.py --public --tunnel-type ngrok
) else (
    python app.py --public --tunnel-type ngrok --ngrok-token="%ngrok_token%"
)

echo.
echo Application stopped.
pause
