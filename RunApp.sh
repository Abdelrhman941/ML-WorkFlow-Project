#!/bin/bash

echo "====================================================="
echo "• Starting ML Studio..."
echo "====================================================="

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not found in PATH"
    echo "Please activate the conda environment first:"
    echo "$ conda activate deploy"
    exit 1
fi

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found in current directory"
    exit 1
fi

# Display Python and environment info
echo "✅ Python found       : $(python --version 2>&1)"
echo "📁 Working directory  : $(pwd)"
echo ""

# Clean session data before starting
echo "====================================================="
echo "• Cleaning session data..."
echo "====================================================="

# Clean session_data folder
if [ -d "session_data" ]; then
    rm -rf session_data/*
    echo "✅ Cleaned session_data folder"
else
    echo "⚠️ session_data folder not found"
fi

echo --------------------------------
# Clean flask_session folder
if [ -d "flask_session" ]; then
    rm -rf flask_session/*
    echo "✅ Cleaned flask_session folder"
else
    echo "⚠️ flask_session folder not found"
fi

echo --------------------------------
echo "⇒ Session cleanup complete!"
echo ""
echo "====================================================="
echo "• Starting Flask application..."
echo "⇒ The application will open in your browser automatically"
echo "⇒ Press Ctrl+C to stop the server"
echo "====================================================="
echo ""

# Function to open browser after Flask is ready
open_browser() {
    # Wait for Flask to be fully ready
    sleep 5
    
    # Wait until Flask server is responding
    URL="http://127.0.0.1:5000"
    MAX_ATTEMPTS=10
    ATTEMPT=0
    
    echo "⏳ Waiting for Flask server to be ready..."
    
    while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
        # Check if server is responding (works in Git Bash on Windows)
        if curl -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null | grep -q "200\|302\|404"; then
            echo "✅ Flask server is ready!"
            echo "🌐 Opening browser at $URL..."
            break
        fi
        ATTEMPT=$((ATTEMPT + 1))
        sleep 1
    done
    
    # Detect OS and open browser accordingly
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        # Windows (Git Bash, MSYS, Cygwin)
        start "$URL" 2>/dev/null || cmd.exe /c start "$URL" 2>/dev/null
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        open "$URL"
    else
        # Linux
        xdg-open "$URL" 2>/dev/null || firefox "$URL" 2>/dev/null || google-chrome "$URL" 2>/dev/null
    fi
}

# Start browser opening in background
open_browser &

# Start the Flask application
python app.py
