# Lumi UI

A full-stack monorepo application with Vue3 frontend and FastAPI backend, managed with pnpm workspaces.

## 🏗️ Project Structure

```
lumi-ui/
├── .env.example                 # Root environment variables template
├── .gitignore                   # Root gitignore (monorepo-wide patterns)
├── package.json                 # Root workspace configuration
├── pnpm-workspace.yaml          # pnpm workspace definition
├── README.md                    # This file
├── apps/
│   ├── client/                  # Vue3 + Vite + Vuetify frontend
│   │   ├── .env.example         # Client-specific environment variables
│   │   ├── .gitignore           # Client-specific gitignore
│   │   ├── package.json         # Client dependencies and scripts
│   │   └── src/                 # Vue application source
│   └── server/                  # FastAPI Python backend
│       ├── .env.example         # Server-specific environment variables
│       ├── .gitignore           # Server-specific gitignore
│       ├── package.json         # Server npm scripts
│       ├── pyproject.toml       # Python dependencies and config
│       ├── main.py              # FastAPI application
│       └── venv/                # Python virtual environment (ignored)
```

## 🚀 Quick Start

### Prerequisites
- Node.js 20.19.0+ or 22.12.0+
- Python 3.11+
- pnpm 10.18.2+

### Installation

1. **Clone and install dependencies:**
   ```bash
   git clone <repository-url>
   cd lumi-ui
   pnpm install
   ```

2. **Set up Python environment:**
   ```bash
   cd apps/server
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install fastapi uvicorn[standard] pydantic python-multipart python-jose[cryptography] passlib[bcrypt] python-dotenv pytest pytest-asyncio httpx ruff
   cd ../..
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   cp apps/client/.env.example apps/client/.env
   cp apps/server/.env.example apps/server/.env
   ```

### Development

**Run both frontend and backend:**
```bash
pnpm dev
```

**Run individually:**
```bash
pnpm dev:client  # Frontend only (http://localhost:5173)
pnpm dev:server  # Backend only (http://localhost:8000)
```

### Building

```bash
pnpm build        # Build all apps
pnpm build:client # Build frontend only
```

### Testing & Linting

```bash
pnpm test         # Run all tests
pnpm lint         # Lint all apps
pnpm test:client  # Test frontend only
pnpm lint:client  # Lint frontend only
```

## 🛠️ Technology Stack

### Frontend (apps/client)
- **Vue 3** - Progressive JavaScript framework
- **Vite** - Fast build tool and dev server
- **Vuetify 3** - Material Design component library
- **TypeScript** - Type-safe JavaScript
- **Pinia** - State management
- **Vue Router** - Client-side routing

### Backend (apps/server)
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Python-JOSE** - JWT handling
- **Passlib** - Password hashing

## 📁 Control Files Organization

### .gitignore Files
- **Root `.gitignore`**: Monorepo-wide patterns (Node.js, Python, IDE files, OS files)
- **Client `.gitignore`**: Frontend-specific patterns (dist, node_modules, build artifacts)
- **Server `.gitignore`**: Python-specific patterns (venv, __pycache__, .ruff_cache)

### Environment Variables
- **Root `.env.example`**: Shared environment variables template
- **Client `.env.example`**: Frontend-specific variables (VITE_ prefixed)
- **Server `.env.example`**: Backend-specific variables (API config, secrets)

## 🔗 API Endpoints

When running the development server, the API is available at `http://localhost:8000`:

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/hello` - Hello world endpoint
- `GET /docs` - Swagger UI documentation
- `GET /redoc` - ReDoc documentation

## 📝 Development Notes

- The monorepo uses pnpm workspaces for dependency management
- Python dependencies are managed per-app with virtual environments
- Both apps run concurrently in development mode with hot reloading
- CORS is configured to allow frontend-backend communication
- All linting and formatting tools are configured and ready to use

## 🤝 Contributing

1. Follow the existing code style and linting rules
2. Add tests for new features
3. Update documentation as needed
4. Use conventional commit messages