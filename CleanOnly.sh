#!/bin/bash

# ML Studio - Clean Session Data Script
# This script removes all files from session_data and flask_session folders

echo "🧹 Cleaning ML Studio session data..."
echo ---------------------------------------

# Clean session_data folder
if [ -d "session_data" ]; then
    rm -rf session_data/*
    echo "✅ Cleaned session_data folder"
else
    echo "⚠️ session_data folder not found"
fi

echo ---------------------------------------

# Clean flask_session folder
if [ -d "flask_session" ]; then
    rm -rf flask_session/*
    echo "✅ Cleaned flask_session folder"
else
    echo "⚠️  flask_session folder not found"
fi

echo ---------------------------------------
# Clean uploads folder (optional - uncomment if needed)
# if [ -d "uploads" ]; then
#     rm -rf uploads/*
#     echo "✅ Cleaned uploads folder"
# fi

echo "✨ Session cleanup complete!"
