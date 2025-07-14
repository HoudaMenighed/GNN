#!/bin/bash

# start_menu.sh - Script to launch the tracktor-mot menu system

# Check if Python is available
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is required but not found."
    exit 1
fi

# Check if the menu file exists
if [ ! -f "tracktor_menu.py" ]; then
    echo "Error: tracktor_menu.py not found."
    exit 1
fi

# Make sure the script is executable
chmod +x tracktor_menu.py

# Launch the menu
echo "Starting Tracktor MOT Menu..."
python3 tracktor_menu.py

# Exit with the same status as the Python script
exit $? 