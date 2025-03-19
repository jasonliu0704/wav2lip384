#!/bin/bash
# Simple wrapper script for smart_cmd.py

# Ensure we're using the right Python
PYTHON_CMD=${PYTHON_CMD:-"python"}

# Find the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the Python script with all arguments passed through
exec $PYTHON_CMD $SCRIPT_DIR/smart_cmd.py "$@"
