#!/bin/bash

# DSPy Prompt Optimizer - Startup Script

echo "🚀 Starting DSPy Prompt Optimizer..."

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Kill existing processes on ports
echo -e "${YELLOW}Cleaning up ports...${NC}"
lsof -ti:8001 | xargs kill -9 2>/dev/null
lsof -ti:3000 | xargs kill -9 2>/dev/null
sleep 1

# Start backend
echo -e "${BLUE}Starting backend server...${NC}"
cd "$SCRIPT_DIR/backend"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv and install deps
source venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null

# Start backend in background
python app.py &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"

# Wait for backend to be ready
sleep 2

# Start frontend
echo -e "${BLUE}Starting frontend server...${NC}"
cd "$SCRIPT_DIR/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

# Start frontend
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"

# Wait for frontend to be ready
sleep 3

# Open browser
echo -e "${BLUE}Opening browser...${NC}"
open http://localhost:3000

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  DSPy Prompt Optimizer is running!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
