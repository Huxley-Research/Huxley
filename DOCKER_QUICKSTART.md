# 🚀 Huxley Docker Quick Start

Get Huxley running in 3 commands:

## Step 1: Start the Container

```bash
docker-compose up
```

This starts:
- 🧬 **Huxley API** (port 8000)
- 🎨 **Interactive Setup UI** (port 3000)
- 📊 **PostgreSQL** database
- ⚡ **Redis** cache

## Step 2: Open the Setup Wizard

Open your browser to **http://localhost:3000**

You'll see an interactive wizard that asks:

### 🤔 "How would you like to run Huxley?"

**Option A: Local Models** 
- Run Llama2, Mistral, etc. locally via Ollama
- No API keys needed
- Requires GPU (recommended: 8GB+ VRAM)

**Option B: Cloud APIs** ☁️
- Use OpenAI (GPT-4), Claude, Gemini, etc.
- No GPU required
- Requires API key from your chosen provider

## Step 3: Configure Your Setup

The wizard guides you through:

1. **Select inference mode** (Local or APIs)
2. **Add API keys** (if using cloud)
3. **Pick your default model**
4. **Configure database** (SQLite for dev, PostgreSQL for production)
5. **Done!** ✅

## Common Setup Paths

### Path 1: Quick Start with OpenAI (Easiest)

1. Select **"Cloud APIs"**
2. Click **"Choose Cloud APIs"**
3. Find **OpenAI** → Enter your API key from https://platform.openai.com/api-keys
4. Select model: **gpt-4**
5. Database: **SQLite** (default, fine for testing)
6. Done! 🎉

### Path 2: Use Intellect-3 via OpenRouter

1. Select **"Cloud APIs"**
2. Find **OpenRouter** → Get key from https://openrouter.ai/keys
3. Select model: **prime-intellect/intellect-3**
4. Rest of setup as above

### Path 3: Local Models with Hardware Detection (No API Keys)

1. First, **Huxley will automatically detect your hardware**:
   - CPU, RAM, GPU, and VRAM
   - Shows efficiency scores for each model
   - Recommends models based on your hardware

2. Install [Ollama](https://ollama.ai) if not installed:
   - Linux: `curl -fsSL https://ollama.ai/install.sh | sh`
   - macOS/Windows: Download from https://ollama.ai

3. In Huxley setup, select **"Local Models"**

4. **Hardware-optimized model recommendations:**
   - **GLM-4.7** (zai-org/GLM-4.7) - 6+ GB VRAM
   - **DeepSeek-V3.2-Speciale** - 10+ GB VRAM  
   - **INTELLECT-3** (PrimeIntellect) - 12+ GB VRAM ⭐
   - **Hermes-4.3-36B** - 40+ GB VRAM
   - **Qwen3-235B-A22B** - 256+ GB VRAM

5. Select a model with **"Excellent"** or **"Good"** efficiency rating

6. Test connection and continue

## After Setup

### 🎯 Try Huxley

```bash
# Interactive chat
docker exec huxley huxley chat

# Generate a protein
docker exec huxley huxley generate -l 100 -d "alpha helix"

# Check system status
docker exec huxley huxley check
```

### 📚 View API Docs

Open http://localhost:8000/docs

### 🛑 Stop the Container

```bash
docker-compose down
```

## Troubleshooting

### "Connection refused" error?
- Make sure containers are running: `docker-compose ps`
- Check logs: `docker-compose logs huxley`

### WebUI not loading?
- Try http://localhost:3000 in an incognito/private browser tab
- Check firewall: Port 3000 must be accessible

### API key not working?
- Double-check the key is copied correctly
- Verify it's from the right provider
- Check the provider account has credits/quota

### Need to restart setup?
```bash
# Clear configuration and start over
docker exec huxley rm ~/.huxley/config.json
docker-compose restart
```

## Next Steps

- 📖 Read the full [Docker guide](DOCKER.md)
- 🔧 Learn about [configuration options](DOCKER.md#configuration-management)
- 🚀 Deploy to production with [advanced setup](DOCKER.md#production-deployment)
- 💬 Ask questions in [discussions](https://github.com/Huxley-Research/Huxley/discussions)

---

**Questions?** Check the [Huxley GitHub](https://github.com/Huxley-Research/Huxley)
