#!/bin/bash

# DSPy Prompt Optimizer - Install all dependencies

set -e

echo "Installing backend dependencies..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

echo "Installing frontend dependencies..."
cd ../frontend
npm install

echo "✓ Done. Run ./start.sh to launch."
