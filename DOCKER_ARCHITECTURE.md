# Huxley Docker Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User's Browser                           │
│                    (http://localhost:3000)                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    WebUI Onboarding
                    (Flask Application)
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │  Setup  │         │  Config │        │   API   │
    │ Wizard  │        │Management│       │ Server  │
    └────┬────┘         └────┬────┘        └────┬────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Huxley Config  │
                    │  (~/.huxley/)   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼─────┐        ┌────▼────┐        ┌────▼────┐
    │PostgreSQL│        │  Redis  │        │  Ollama  │
    │ Database │        │  Cache  │        │ (Optional)│
    └──────────┘        └─────────┘        └──────────┘
```

## Docker Compose Setup

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network                             │
│                   (huxley-network)                            │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                  Huxley Service                       │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │  WebUI Onboarding (Flask)       [Port 3000]    │ │   │
│  │  │  - Interactive setup wizard                     │ │   │
│  │  │  - Configuration management                    │ │   │
│  │  │  - REST API endpoints                          │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │  Huxley API Server (FastAPI)   [Port 8000]     │ │   │
│  │  │  - REST API /docs                              │ │   │
│  │  │  - LLM operations                              │ │   │
│  │  │  - Tool registry                               │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────────┐ │   │
│  │  │  Configuration Volume                          │ │   │
│  │  │  ~/.huxley/                                     │ │   │
│  │  │  ├── config.json                               │ │   │
│  │  │  └── huxley.db (SQLite)                        │ │   │
│  │  └─────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────┐  ┌────────────────────────┐   │
│  │   PostgreSQL Service     │  │   Redis Service        │   │
│  │   [Port 5432]            │  │   [Port 6379]          │   │
│  │   ├── postgres_data vol  │  │   ├── redis_data vol   │   │
│  │   └── huxley database    │  │   └── cache store      │   │
│  └──────────────────────────┘  └────────────────────────┘   │
│                                                               │
│  ┌──────────────────────────┐ (Optional)                     │
│  │    Ollama Service        │                                │
│  │   [Port 11434]           │                                │
│  │   └── ollama_data vol    │                                │
│  └──────────────────────────┘                                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
        ▲
        │ Host System
        │
┌───────▼────────────────────────────────────────────────────┐
│  Ports Exposed                                              │
│  - 3000:3000   (WebUI Onboarding)                           │
│  - 8000:8000   (API Server)                                 │
│  - 5432:5432   (PostgreSQL)                                 │
│  - 6379:6379   (Redis)                                      │
│  - 11434:11434 (Ollama - if enabled)                        │
└─────────────────────────────────────────────────────────────┘
```

## Onboarding Flow

```
Start: docker-compose up
        │
        ▼
   ┌─────────────────┐
   │  Check Config   │
   └────┬────────────┘
        │
    ┌───┴───────────────────────┐
    │                           │
    │ Config exists?            │
    │                           │
   YES                         NO
    │                           │
    ▼                           ▼
Start API Server      Start WebUI Onboarding
(port 8000)          (port 3000)
    │                           │
    │                    ┌──────┴──────┐
    │                    │             │
    │              Step 1: Mode        │
    │          Local or APIs?          │
    │          (User Choice)           │
    │                    │             │
    │          ┌─────────┴─────────┐   │
    │          │                   │   │
    │        Local               APIs  │
    │          │                   │   │
    │          ▼                   ▼   │
    │      Config         Config Providers
    │      Ollama         (OpenAI, etc)
    │          │                   │   │
    │          └─────────┬─────────┘   │
    │                    │             │
    │              Step 2: Select
    │                 Model
    │                    │             │
    │          ┌─────────┴─────────┐   │
    │          │                   │   │
    │     Ollama models     API Models  │
    │          │                   │   │
    │          └─────────┬─────────┘   │
    │                    │             │
    │              Step 3: Database
    │             (SQLite/PostgreSQL)
    │                    │             │
    │                    ▼             │
    │          Config saved            │
    │          in ~/.huxley/config.json│
    │                    │             │
    │                    ▼             │
    │          Setup Complete!         │
    │                    │             │
    │                    └─────────────┘
    │
    ▼
Ready to Use
API: http://localhost:8000
Docs: http://localhost:8000/docs
```

## Configuration Storage

```
~/.huxley/
├── config.json              (Main configuration)
│   ├── setup_complete: boolean
│   ├── inference_mode: "api" | "local"
│   ├── default_provider: string
│   ├── default_model: string
│   ├── providers_configured: [string, ...]
│   ├── providers: {
│   │   "provider_id": {
│   │       "api_key": string,
│   │       "configured_at": timestamp
│   │   }
│   │}
│   └── database: {
│       "driver": "sqlite" | "postgresql",
│       "host": string,
│       "port": number,
│       "database": string
│   }
│
├── huxley.db                (SQLite database - if driver=sqlite)
│
└── .env                     (Environment variables)
    ├── OPENAI_API_KEY
    ├── ANTHROPIC_API_KEY
    └── ...
```

## Data Flow

### During Onboarding

```
User Input (WebUI)
        │
        ▼
JavaScript Frontend (app.js)
        │
        ▼
Flask REST API (onboarding.py)
        │
        ├─ Validate input
        ├─ Store in ConfigurationManager
        │
        ▼
config.json (persistent storage)
        │
        └─ Ready for API Server startup
```

### During Operation

```
API Request (port 8000)
        │
        ▼
FastAPI Handler
        │
        ├─ Load config.json
        ├─ Get provider credentials
        ├─ Connect to model
        │
        └─ PostgreSQL / Redis as needed
                │
                └─ Store history/cache
```

## Provider Integration

```
Huxley Docker
        │
        ├─ Local Models
        │  └─ Ollama Service (container)
        │     ├─ Llama2
        │     ├─ Mistral
        │     └─ Neural Chat
        │
        └─ Cloud APIs
           ├─ OpenAI (gpt-4, etc)
           ├─ Anthropic (Claude)
           ├─ Google (Gemini)
           ├─ Cohere (Command)
           ├─ OpenRouter
           │  └─ Intellect-3 ⭐ (NEW)
           └─ Together.ai
```

## Security Model

```
┌─────────────────────────────────────────────┐
│         Huxley Container                     │
│         (Running as huxley user)            │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │ Configuration & Secrets                │ │
│  │ ~/.huxley/config.json                  │ │
│  │ - API keys stored encrypted            │ │
│  │ - Permissions: 0600 (owner only)       │ │
│  └────────────────────────────────────────┘ │
│                                              │
│  ┌────────────────────────────────────────┐ │
│  │ Running Application                    │ │
│  │ - Non-root user (huxley, UID 1000)     │ │
│  │ - Limited file system access           │ │
│  │ - Network isolation via docker networks│ │
│  └────────────────────────────────────────┘ │
│                                              │
└─────────────────────────────────────────────┘
        │
        └─ External Services
           ├─ Database (PostgreSQL)
           │  └─ Password protected
           ├─ Cache (Redis)
           │  └─ Local network only
           └─ API Providers (HTTPS)
              └─ Encrypted in transit
```

## Multi-Provider Model Selection

```
Huxley Auto Selector
        │
        ├─ Available Providers (from config)
        │
        ├─ Filter by capabilities
        │  ├─ Vision support?
        │  ├─ Reasoning mode?
        │  └─ Tool use?
        │
        ├─ Filter by cost preference
        │  ├─ Economy
        │  ├─ Balanced
        │  └─ Performance
        │
        └─ Rank by task fit
           ├─ Reasoning tasks
           ├─ Coding tasks
           ├─ Creative tasks
           └─ Fast tasks
                │
                └─ Select best model

Intellect-3 Capabilities:
    - Reasoning: 92/100 ⭐⭐⭐
    - Coding: 88/100 ⭐⭐⭐
    - Speed: 60/100 ⭐⭐
    - Cost: 42/100 (optimized pricing)
    ═══════════════════════
    - Best for: Research, Coding, Analysis
```

This architecture ensures:
✅ User-friendly onboarding
✅ Multiple inference options
✅ Secure credential storage
✅ Scalable database options
✅ Flexible provider support
✅ Production-ready deployment
