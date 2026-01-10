# Docker Setup - Complete Reference

## 📋 Quick Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| **[DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)** | Start here! 3-step setup | Everyone |
| **[DOCKER.md](DOCKER.md)** | Full guide & reference | Users & Admins |
| **[DOCKER_ARCHITECTURE.md](DOCKER_ARCHITECTURE.md)** | System design & diagrams | Developers |
| **[DOCKER_IMPLEMENTATION.md](DOCKER_IMPLEMENTATION.md)** | Technical details | Contributors |
| **[DOCKER_SETUP_COMPLETE.md](DOCKER_SETUP_COMPLETE.md)** | What was built | Project Leads |

---

## 🚀 Start Here

### 3-Command Setup

```bash
# 1. Navigate to Huxley
cd /workspaces/Huxley

# 2. Start the Docker stack
docker-compose up

# 3. Open your browser
# http://localhost:3000
```

That's it! The interactive onboarding will guide you through the rest.

---

## 📁 What's New

### Docker Files
```
Dockerfile                    # Multi-stage build
docker-compose.yml           # Full stack definition
docker-entrypoint.sh         # Smart startup script
.dockerignore                # Build optimization
```

### Huxley Docker Module
```
src/huxley/docker/
├── __init__.py              # Package marker
├── __main__.py              # Entry point
├── onboarding.py            # Flask app (REST API)
├── templates/index.html     # Web interface
└── static/
    ├── style.css            # Styling
    └── app.js               # Client logic
```

### Documentation
```
DOCKER_QUICKSTART.md         # ⭐ Start here
DOCKER.md                    # Complete guide
DOCKER_ARCHITECTURE.md       # System design
DOCKER_IMPLEMENTATION.md     # Technical details
DOCKER_SETUP_COMPLETE.md     # Summary
```

---

## 🎯 Key Features

### Interactive Onboarding
- No command-line knowledge needed
- Beautiful, responsive web interface
- Step-by-step wizard (6 steps)
- Input validation & error handling

### Dual Inference Modes
- **Local**: Ollama (privacy, no API key)
- **Cloud APIs**: OpenAI, Claude, Gemini, Intellect-3, and more

### Multi-Provider Support
Select from 6+ providers during setup:
- OpenAI (GPT-4, GPT-4 Turbo)
- Anthropic (Claude)
- Google (Gemini)
- Cohere (Command)
- **OpenRouter** (Intellect-3 ⭐)
- Together.ai

### Production Ready
- Proper database options (SQLite or PostgreSQL)
- Redis caching
- Health checks
- Service dependencies
- Persistent volumes

---

## 📊 Architecture Overview

### Container Stack
```
WebUI Onboarding (Port 3000)
    ↓
Huxley API Server (Port 8000)
    ↓
PostgreSQL (Port 5432)
Redis (Port 6379)
Ollama (Port 11434 - optional)
```

### Configuration Flow
```
User Input (WebUI)
    ↓
REST API Handler
    ↓
ConfigurationManager
    ↓
~/.huxley/config.json (persistent)
    ↓
API Server (next run)
```

---

## 🔧 Usage Examples

### Start Everything
```bash
docker-compose up
```

### Start Specific Services
```bash
docker-compose up huxley
```

### View Logs
```bash
docker-compose logs -f huxley
```

### Execute Commands
```bash
docker exec huxley huxley chat
docker exec huxley huxley generate -l 100
```

### Reset Configuration
```bash
docker exec huxley rm ~/.huxley/config.json
docker-compose restart
```

### Stop Everything
```bash
docker-compose down
```

### Clean Everything (Including Data)
```bash
docker-compose down -v
```

---

## 🎓 Setup Paths

### Path 1: OpenAI (Easiest)
1. `docker-compose up`
2. Open http://localhost:3000
3. Select "Cloud APIs"
4. OpenAI → Add key
5. Select gpt-4
6. Done!

### Path 2: Intellect-3 via OpenRouter
1. `docker-compose up`
2. Open http://localhost:3000
3. Select "Cloud APIs"
4. OpenRouter → Add key
5. Select intellect-3
6. Done!

### Path 3: Local Llama2
1. Install Ollama
2. `ollama pull llama2`
3. `docker-compose up`
4. Open http://localhost:3000
5. Select "Local Models"
6. Done!

---

## 📖 Documentation Map

### For Users
- **Quick Start**: [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)
- **Full Guide**: [DOCKER.md](DOCKER.md)
- **Troubleshooting**: See [DOCKER.md](DOCKER.md#troubleshooting)

### For Developers
- **Architecture**: [DOCKER_ARCHITECTURE.md](DOCKER_ARCHITECTURE.md)
- **Implementation**: [DOCKER_IMPLEMENTATION.md](DOCKER_IMPLEMENTATION.md)
- **Code**: `src/huxley/docker/`

### For DevOps
- **Docker Compose**: `docker-compose.yml`
- **Build**: `Dockerfile`
- **Startup**: `docker-entrypoint.sh`
- **Deployment**: [DOCKER.md - Production](DOCKER.md#production-deployment)

---

## 🔌 API Endpoints

### Onboarding API (Port 3000)
```
GET  /health                           # Health check
GET  /api/status                       # Setup status
GET  /api/config                       # Current config
GET  /api/providers                    # Available providers
POST /api/setup/inference-mode         # Set mode
POST /api/setup/configure-provider     # Add provider
POST /api/setup/select-model           # Choose model
POST /api/setup/database               # Configure DB
POST /api/setup/complete               # Finish setup
POST /api/test-connection              # Test service
```

### Huxley API (Port 8000)
```
GET  /docs                    # Swagger UI
GET  /redoc                   # ReDoc
*    /api/...                 # Huxley API (after setup)
```

---

## ⚙️ Environment Variables

Configure via `docker-compose.yml` or `.env`:

```bash
# Server
HUXLEY_SERVER_HOST=0.0.0.0
HUXLEY_SERVER_PORT=8000
HUXLEY_ENV=production
HUXLEY_DEBUG=false

# WebUI
WEBUI_HOST=0.0.0.0
WEBUI_PORT=3000

# Database
HUXLEY_DB_DRIVER=postgresql    # or sqlite
HUXLEY_DB_HOST=postgres
HUXLEY_DB_PORT=5432
HUXLEY_DB_USERNAME=huxley
HUXLEY_DB_PASSWORD=password
HUXLEY_DB_DATABASE=huxley

# Redis
HUXLEY_REDIS_HOST=redis
HUXLEY_REDIS_PORT=6379
```

---

## 🔐 Security

- ✅ Non-root Docker user (UID 1000)
- ✅ Configuration isolation
- ✅ Encrypted credentials
- ✅ Network isolation
- ✅ Health checks
- ✅ Proper secrets handling

---

## 📚 Integration with Intellect-3

The Docker setup includes full support for Intellect-3:

### In Configuration
- ✅ Added to OpenRouter provider list
- ✅ Listed in provider setup options

### In Prompts
- ✅ Specialized prompt template for Intellect-3
- ✅ Optimized for reasoning and analysis

### In Auto-Selector
- ✅ Model capabilities defined
- ✅ Perfect for reasoning/coding/research tasks

Users can select it during setup:
```
Provider: OpenRouter
Model: prime-intellect/intellect-3
API Key: sk-or-... (from https://openrouter.ai/keys)
```

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| WebUI won't load | Check port 3000 is open, verify container is running |
| API key error | Verify key is correct, account has credits |
| Database connection fails | Check PostgreSQL is running: `docker-compose ps` |
| Configuration lost | Use volume persistence: `docker volume ls` |
| Need to reset | `docker exec huxley rm ~/.huxley/config.json` |

See [DOCKER.md - Troubleshooting](DOCKER.md#troubleshooting) for detailed solutions.

---

## 🚀 Next Steps

1. **Read**: [DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md) (5 min read)
2. **Start**: `docker-compose up`
3. **Setup**: Follow the interactive wizard
4. **Use**: Access http://localhost:8000/docs
5. **Explore**: Try `docker exec huxley huxley chat`

---

## 📞 Support

- 📖 Full docs: [DOCKER.md](DOCKER.md)
- 🐛 Issues: https://github.com/Huxley-Research/Huxley/issues
- 💬 Discussions: https://github.com/Huxley-Research/Huxley/discussions

---

## 📋 Checklist

- [x] Dockerfile created
- [x] docker-compose.yml configured
- [x] WebUI onboarding built
- [x] REST API endpoints implemented
- [x] Configuration management system
- [x] Multi-provider support
- [x] Local model support (Ollama)
- [x] Database options (SQLite/PostgreSQL)
- [x] Comprehensive documentation
- [x] Architecture diagrams
- [x] Quick start guide
- [x] Intellect-3 integration

**Everything is ready to go!** 🎉

---

**Last Updated**: January 10, 2026  
**Version**: Huxley 0.6.0 with Docker
