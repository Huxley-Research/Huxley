# Huxley Docker Setup Guide

This guide explains how to run Huxley using Docker with an interactive WebUI onboarding experience.

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Start Huxley with all services (API, WebUI, PostgreSQL, Redis)
docker-compose up

# In another terminal, open the onboarding UI
open http://localhost:3000
```

The Docker Compose setup includes:
- **Huxley API Server** on http://localhost:8000
- **WebUI Onboarding** on http://localhost:3000
- **PostgreSQL** for persistent storage
- **Redis** for caching and task distribution

### Option 2: Using Docker Directly

```bash
# Build the image
docker build -t huxley:latest .

# Run the container
docker run -d \
  -p 3000:3000 \
  -p 8000:8000 \
  -v ~/.huxley:/home/huxley/.huxley \
  --name huxley \
  huxley:latest

# Access the onboarding UI
open http://localhost:3000
```

## WebUI Onboarding Flow

When you first start Huxley, the interactive onboarding wizard will guide you through:

### 1. **Choose Inference Mode**
   - **Local Models**: Run models directly using Ollama (requires GPU)
   - **Cloud APIs**: Use OpenAI, Claude, Gemini, Cohere, OpenRouter, etc.

### 2. **Configure Providers (if using APIs)**
   - Add API keys for your preferred providers
   - Support for multiple providers simultaneously
   - Secure key storage in encrypted configuration

### 3. **Select Default Model**
   - Choose your primary model from configured providers
   - Available models depend on selected provider
   - Examples:
     - OpenAI: gpt-4, gpt-4-turbo
     - Claude: claude-opus, claude-sonnet
     - Intellect-3 via OpenRouter: prime-intellect/intellect-3

### 4. **Setup Local Models (Optional)**
   - If using Ollama, configure connection settings
   - Test Ollama connectivity
   - Select preferred models (Llama2, Mistral, Neural Chat, etc.)

### 5. **Configure Database**
   - **SQLite**: Good for development, single machine
   - **PostgreSQL**: Recommended for production, multiple instances
   - Credentials and connection details

### 6. **Completion**
   - Setup complete! Ready to use Huxley
   - Links to API documentation
   - Next steps for first use

## Configuration Management

After initial setup, your configuration is stored in:

```
~/.huxley/config.json          # Main configuration
~/.huxley/.env                  # Environment variables
~/.huxley/huxley.db             # SQLite database (if using SQLite)
```

### Configuration Structure

```json
{
  "setup_complete": true,
  "inference_mode": "api",
  "default_provider": "openai",
  "default_model": "gpt-4",
  "providers_configured": ["openai", "anthropic"],
  "providers": {
    "openai": {
      "api_key": "sk-...",
      "configured_at": "2024-01-10T12:00:00"
    }
  },
  "database": {
    "driver": "postgresql",
    "host": "postgres",
    "port": 5432,
    "database": "huxley"
  }
}
```

## Environment Variables

Configure Huxley via environment variables in Docker:

```bash
# Server Configuration
HUXLEY_SERVER_HOST=0.0.0.0
HUXLEY_SERVER_PORT=8000
HUXLEY_ENV=production
HUXLEY_DEBUG=false

# WebUI Configuration
WEBUI_HOST=0.0.0.0
WEBUI_PORT=3000

# Database Configuration (PostgreSQL)
HUXLEY_DB_DRIVER=postgresql
HUXLEY_DB_HOST=postgres
HUXLEY_DB_PORT=5432
HUXLEY_DB_USERNAME=huxley
HUXLEY_DB_PASSWORD=your_password
HUXLEY_DB_DATABASE=huxley

# Redis Configuration
HUXLEY_REDIS_HOST=redis
HUXLEY_REDIS_PORT=6379

# API Keys (set in onboarding, or pre-configure here)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENROUTER_API_KEY=sk-or-...
```

## Advanced Usage

### Custom Docker Image

Create a custom Dockerfile with pre-configured API keys:

```dockerfile
FROM huxley:latest

# Copy pre-configured setup
COPY config.json ~/.huxley/config.json

# Skip onboarding on startup
ENV HUXLEY_SKIP_ONBOARDING=true
```

### Local Model Support (Ollama)

To use local models, uncomment the Ollama service in `docker-compose.yml`:

```yaml
ollama:
  image: ollama/ollama:latest
  container_name: huxley-ollama
  ports:
    - "11434:11434"
  volumes:
    - ollama_data:/root/.ollama
```

Then in the onboarding, select "Local Models" and configure Ollama connection.

### Production Deployment

For production deployments:

1. **Use environment secrets** instead of storing keys in config files:
   ```bash
   docker run -e OPENAI_API_KEY=$(cat /run/secrets/openai_key) ...
   ```

2. **Enable HTTPS** on the API server:
   ```bash
   HUXLEY_SERVER_SSL_CERT=/path/to/cert.pem \
   HUXLEY_SERVER_SSL_KEY=/path/to/key.pem
   ```

3. **Use external PostgreSQL** for better reliability:
   ```bash
   HUXLEY_DB_HOST=production-postgres.example.com
   ```

4. **Set up Redis replication** for distributed caching

### Troubleshooting

#### Onboarding UI not accessible
- Check that port 3000 is exposed: `docker port huxley`
- Verify container is running: `docker ps | grep huxley`
- Check logs: `docker logs huxley`

#### API server won't start after onboarding
- Verify configuration: `cat ~/.huxley/config.json`
- Check database connectivity: `docker logs huxley-postgres`
- Review API logs: `docker logs huxley -f`

#### Models not responding
- If using OpenAI: Verify API key is valid
- If using Ollama: Ensure Ollama container is running
- Check network connectivity between containers: `docker network inspect huxley-network`

## API Documentation

Once setup is complete, access the API documentation:

```
http://localhost:8000/docs        # Swagger UI
http://localhost:8000/redoc       # ReDoc
```

## Next Steps

1. **Try the Chat Interface**:
   ```bash
   docker exec huxley huxley chat
   ```

2. **Generate a Protein**:
   ```bash
   docker exec huxley huxley generate -l 100 -d "alpha helix"
   ```

3. **Read the Documentation**:
   - Full documentation: https://github.com/Huxley-Research/Huxley
   - API Reference: http://localhost:8000/docs

## Support

For issues or questions:
- GitHub: https://github.com/Huxley-Research/Huxley/issues
- Discussions: https://github.com/Huxley-Research/Huxley/discussions
