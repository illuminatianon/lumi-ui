# Lumi Server

FastAPI backend server for the Lumi UI application.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e .
```

3. For development dependencies:
```bash
pip install -e ".[dev]"
```

4. Copy environment file:
```bash
cp .env.example .env
```

## Development

Run the development server:
```bash
pnpm dev
```

Or directly with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Testing

### Unit and Integration Tests
Run the test suite:
```bash
pnpm test
```

### Manual Testing
For testing specific functionality:
- **Inference Infrastructure**: See [Testing Guide](docs/testing-guide.md)
- **Image Generation**: See [Image Testing Guide](docs/image-testing-guide.md)

Test files are organized in the `tests/` directory:
- `tests/unit/` - Unit tests
- `tests/integration/` - Integration tests
- `tests/manual/` - Manual test scripts

## Linting and Formatting

```bash
pnpm lint
pnpm format
```
