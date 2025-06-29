#!/bin/bash

# Quran Detection App - Public Access Starter
# This script helps you run the app with public tunnel access

echo "🕌 Quran Verse Detection - Public Access Setup"
echo "=============================================="

# Check if Python is installed
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Use python3 if available, otherwise python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo "❌ pip is not installed. Please install pip."
    exit 1
fi

# Use pip3 if available, otherwise pip
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
else
    PIP_CMD="pip"
fi

echo "📦 Using pip: $PIP_CMD"

# Install required packages
echo ""
echo "📥 Installing required Python packages..."
$PIP_CMD install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install requirements. Please check the error above."
    exit 1
fi

echo "✅ Python packages installed successfully!"

# Check for optional tunnel tools
echo ""
echo "🔧 Checking available tunnel options..."

# Check for ngrok token
if [ -z "$NGROK_AUTH_TOKEN" ]; then
    echo "⚠️  NGROK_AUTH_TOKEN not set. You can get a free token from https://ngrok.com"
else
    echo "✅ Ngrok auth token found"
fi

# Check for Node.js and localtunnel
if command -v npm &> /dev/null; then
    echo "✅ npm found - LocalTunnel available"
    echo "   Installing localtunnel globally..."
    npm install -g localtunnel 2>/dev/null || echo "⚠️  Failed to install localtunnel globally"
else
    echo "⚠️  npm not found - LocalTunnel not available"
    echo "   Install Node.js from https://nodejs.org/ to use LocalTunnel"
fi

# Check for cloudflared
if command -v cloudflared &> /dev/null; then
    echo "✅ cloudflared found - Cloudflare Tunnel available"
else
    echo "⚠️  cloudflared not found - Cloudflare Tunnel not available"
    echo "   Download from https://github.com/cloudflare/cloudflared/releases"
fi

# Check for SSH (for Serveo)
if command -v ssh &> /dev/null; then
    echo "✅ SSH found - Serveo tunnel available"
else
    echo "⚠️  SSH not found - Serveo tunnel not available"
fi

echo ""
echo "🚀 Starting application with public access..."
echo ""

# Parse command line arguments
TUNNEL_TYPE="auto"
NGROK_TOKEN=""
LOCAL_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --local)
            LOCAL_ONLY=true
            shift
            ;;
        --tunnel-type)
            TUNNEL_TYPE="$2"
            shift 2
            ;;
        --ngrok-token)
            NGROK_TOKEN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --local              Run in local mode only (no public tunnel)"
            echo "  --tunnel-type TYPE   Specify tunnel type (auto, ngrok, localtunnel, cloudflare, serveo)"
            echo "  --ngrok-token TOKEN  Specify ngrok auth token"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Auto-detect best tunnel"
            echo "  $0 --local                           # Local access only"
            echo "  $0 --tunnel-type ngrok               # Force ngrok tunnel"
            echo "  $0 --tunnel-type ngrok --ngrok-token YOUR_TOKEN"
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set environment variables
if [ ! -z "$NGROK_TOKEN" ]; then
    export NGROK_AUTH_TOKEN="$NGROK_TOKEN"
fi

# Run the application
if [ "$LOCAL_ONLY" = true ]; then
    echo "🏠 Starting in local mode only..."
    echo "🌐 Access your app at: http://127.0.0.1:5000"
    echo "📱 Manage public access from: http://127.0.0.1:5000/tunnel"
    echo ""
    $PYTHON_CMD app.py
else
    echo "🌍 Starting with public tunnel (type: $TUNNEL_TYPE)..."
    echo ""
    
    if [ ! -z "$NGROK_TOKEN" ]; then
        $PYTHON_CMD app.py --public --tunnel-type "$TUNNEL_TYPE" --ngrok-token "$NGROK_TOKEN"
    else
        $PYTHON_CMD app.py --public --tunnel-type "$TUNNEL_TYPE"
    fi
fi
