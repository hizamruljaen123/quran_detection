#!/bin/bash

echo "Starting Quran Verse Detection Web Application..."
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found!"
    echo "Please run this script from the Web directory."
    exit 1
fi

# Check if requirements are installed
echo "Checking dependencies..."
python3 -c "import flask, numpy, librosa, tensorflow, sklearn, mysql.connector" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies!"
        exit 1
    fi
fi

# Create necessary directories
mkdir -p static/uploads
mkdir -p models

echo
echo "Starting Flask application..."
echo "Application will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo

python3 app.py
