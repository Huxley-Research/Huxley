#!/bin/bash
# Huxley Docker Startup Script
# Handles both onboarding and normal operation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🧬 Huxley - Biological Computational Engine${NC}"
echo "=================================="
echo ""

# Check if setup is already complete
if [ -f "$HOME/.huxley/config.json" ]; then
    CONFIG=$(cat $HOME/.huxley/config.json)
    if grep -q '"setup_complete": true' <<< "$CONFIG"; then
        echo -e "${GREEN}✓${NC} Huxley is already configured"
        echo "Starting API server..."
        exec python -m uvicorn huxley.api.app:app \
            --host ${HUXLEY_SERVER_HOST:-0.0.0.0} \
            --port ${HUXLEY_SERVER_PORT:-8000} \
            --reload
    fi
fi

# Setup is not complete, start onboarding
echo -e "${YELLOW}ℹ${NC} First-time setup detected"
echo "Starting interactive onboarding WebUI..."
echo ""
echo -e "Open your browser to: ${BLUE}http://localhost:3000${NC}"
echo ""

exec python -m huxley.docker.onboarding
