#!/bin/bash
# Run backend tests

echo "Running backend tests..."

# Set PYTHONPATH to include the current directory
export PYTHONPATH="${PYTHONPATH}:."

# Set DYLD_LIBRARY_PATH for Opus library on macOS
export DYLD_LIBRARY_PATH=/opt/homebrew/lib:$DYLD_LIBRARY_PATH

# Run tests
pytest tests/ -v

echo "Tests completed!"