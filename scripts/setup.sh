#!/bin/bash

# ===================================
# Lumi UI Setup Script
# ===================================

set -e

echo "🚀 Setting up Lumi UI monorepo..."

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 20.19.0+ or 22.12.0+"
    exit 1
fi

if ! command -v pnpm &> /dev/null; then
    echo "❌ pnpm is not installed. Please install pnpm 10.18.2+"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11+"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
pnpm install

# Set up Python environment
echo "🐍 Setting up Python environment..."
cd apps/server

if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Installing Python dependencies..."
source venv/bin/activate
pip install fastapi uvicorn[standard] pydantic python-multipart python-jose[cryptography] passlib[bcrypt] python-dotenv pytest pytest-asyncio httpx ruff

cd ../..

# Set up environment files
echo "⚙️  Setting up environment files..."

if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created root .env file"
fi

if [ ! -f "apps/client/.env" ]; then
    cp apps/client/.env.example apps/client/.env
    echo "✅ Created client .env file"
fi

if [ ! -f "apps/server/.env" ]; then
    cp apps/server/.env.example apps/server/.env
    echo "✅ Created server .env file"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start development:"
echo "  pnpm dev"
echo ""
echo "Individual commands:"
echo "  pnpm dev:client  # Frontend only"
echo "  pnpm dev:server  # Backend only"
echo ""
echo "Other commands:"
echo "  pnpm build       # Build all"
echo "  pnpm test        # Test all"
echo "  pnpm lint        # Lint all"
