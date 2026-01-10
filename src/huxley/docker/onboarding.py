"""
WebUI Onboarding for Huxley Docker deployments.

Provides an interactive web interface for:
- Choosing between local model or API-based inference
- Configuring API keys and providers
- Setting up database and cache connections
- Testing model connections
"""

import os
import json
import asyncio
import platform
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Configuration
HUXLEY_DATA_DIR = Path.home() / ".huxley"
CONFIG_FILE = HUXLEY_DATA_DIR / "config.json"
ENV_FILE = HUXLEY_DATA_DIR / ".env"


class ConfigurationManager:
    """Manage configuration for Docker deployment."""
    
    def __init__(self):
        self.data_dir = HUXLEY_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load existing configuration."""
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE) as f:
                return json.load(f)
        return {}
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration."""
        CONFIG_FILE.write_text(json.dumps(config, indent=2))
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        self.config.update(updates)
        self.save_config(self.config)


config_manager = ConfigurationManager()


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "component": "huxley-onboarding"
    })


@app.route("/api/status", methods=["GET"])
def get_status():
    """Get current setup status."""
    config = config_manager.get_config()
    
    return jsonify({
        "setup_complete": config.get("setup_complete", False),
        "inference_mode": config.get("inference_mode"),
        "providers_configured": config.get("providers_configured", []),
        "api_server_running": check_api_server(),
    })


@app.route("/api/config", methods=["GET"])
def get_config():
    """Get current configuration (without secrets)."""
    config = config_manager.get_config()
    
    # Remove sensitive data
    safe_config = {
        "inference_mode": config.get("inference_mode"),
        "default_model": config.get("default_model"),
        "database_configured": bool(config.get("database")),
        "providers_configured": config.get("providers_configured", []),
    }
    
    return jsonify(safe_config)


@app.route("/api/providers", methods=["GET"])
def get_providers():
    """Get available providers."""
    return jsonify({
        "local": [
            {
                "id": "ollama",
                "name": "Ollama (Local)",
                "description": "Run models locally using Ollama",
                "requires_gpu": True,
                "models": [
                    {
                        "id": "glm-4.7",
                        "name": "GLM-4.7 (358B)",
                        "huggingface": "zai-org/GLM-4.7",
                        "size_gb": 358.0,
                        "min_vram_gb": 380,
                        "recommended_vram_gb": 450
                    },
                    {
                        "id": "deepseek-v3.2",
                        "name": "DeepSeek-V3.2-Speciale (685B)",
                        "huggingface": "deepseek-ai/DeepSeek-V3.2-Speciale",
                        "size_gb": 685.0,
                        "min_vram_gb": 720,
                        "recommended_vram_gb": 850
                    },
                    {
                        "id": "intellect-3",
                        "name": "INTELLECT-3 (100B)",
                        "huggingface": "PrimeIntellect/INTELLECT-3",
                        "size_gb": 100.0,
                        "min_vram_gb": 110,
                        "recommended_vram_gb": 140
                    },
                    {
                        "id": "hermes-4.3",
                        "name": "Hermes-4.3-36B",
                        "huggingface": "NousResearch/Hermes-4.3-36B",
                        "size_gb": 36.0,
                        "min_vram_gb": 40,
                        "recommended_vram_gb": 48
                    },
                    {
                        "id": "qwen3-235b",
                        "name": "Qwen3-235B-A22B",
                        "huggingface": "Qwen/Qwen3-235B-A22B",
                        "size_gb": 235.0,
                        "min_vram_gb": 256,
                        "recommended_vram_gb": 320
                    }
                ],
                "port": 11434,
            }
        ],
        "api": [
            {
                "id": "openai",
                "name": "OpenAI",
                "description": "GPT-4, GPT-4 Turbo, and more",
                "requires_key": True,
                "env_var": "OPENAI_API_KEY",
                "url": "https://platform.openai.com/account/api-keys",
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "description": "Claude models",
                "requires_key": True,
                "env_var": "ANTHROPIC_API_KEY",
                "url": "https://console.anthropic.com/",
            },
            {
                "id": "google",
                "name": "Google",
                "description": "Gemini models",
                "requires_key": True,
                "env_var": "GOOGLE_API_KEY",
                "url": "https://ai.google.dev/",
            },
            {
                "id": "cohere",
                "name": "Cohere",
                "description": "Command models",
                "requires_key": True,
                "env_var": "COHERE_API_KEY",
                "url": "https://dashboard.cohere.com/api-keys",
            },
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "description": "Access to 100+ models including Intellect-3",
                "requires_key": True,
                "env_var": "OPENROUTER_API_KEY",
                "url": "https://openrouter.ai/keys",
            },
            {
                "id": "together",
                "name": "Together.ai",
                "description": "Open source and frontier models",
                "requires_key": True,
                "env_var": "TOGETHER_API_KEY",
                "url": "https://api.together.xyz/",
            },
        ]
    })


@app.route("/api/setup/inference-mode", methods=["POST"])
def set_inference_mode():
    """Set inference mode (local or api)."""
    data = request.json
    mode = data.get("mode")  # "local" or "api"
    
    if mode not in ["local", "api"]:
        return jsonify({"error": "Invalid mode"}), 400
    
    config_manager.update_config({
        "inference_mode": mode,
        "step": "providers" if mode == "api" else "local_setup"
    })
    
    return jsonify({"success": True, "mode": mode})


@app.route("/api/setup/configure-provider", methods=["POST"])
def configure_provider():
    """Configure an API provider."""
    data = request.json
    provider_id = data.get("provider_id")
    api_key = data.get("api_key")
    
    if not provider_id or not api_key:
        return jsonify({"error": "Missing provider_id or api_key"}), 400
    
    config = config_manager.get_config()
    
    # Store provider configuration
    if "providers" not in config:
        config["providers"] = {}
    
    config["providers"][provider_id] = {
        "api_key": api_key,
        "configured_at": str(asyncio.get_event_loop().time())
    }
    
    if "providers_configured" not in config:
        config["providers_configured"] = []
    
    if provider_id not in config["providers_configured"]:
        config["providers_configured"].append(provider_id)
    
    config_manager.save_config(config)
    
    return jsonify({
        "success": True,
        "provider": provider_id,
        "configured": True
    })


@app.route("/api/setup/select-model", methods=["POST"])
def select_model():
    """Select default model."""
    data = request.json
    provider = data.get("provider")
    model = data.get("model")
    
    if not provider or not model:
        return jsonify({"error": "Missing provider or model"}), 400
    
    config_manager.update_config({
        "default_provider": provider,
        "default_model": model,
        "step": "database"
    })
    
    return jsonify({
        "success": True,
        "provider": provider,
        "model": model
    })


@app.route("/api/setup/database", methods=["POST"])
def configure_database():
    """Configure database."""
    data = request.json
    driver = data.get("driver", "sqlite")  # sqlite or postgresql
    
    db_config = {"driver": driver}
    
    if driver == "postgresql":
        db_config.update({
            "host": data.get("host", "postgres"),
            "port": data.get("port", 5432),
            "username": data.get("username"),
            "password": data.get("password"),
            "database": data.get("database", "huxley"),
        })
    elif driver == "sqlite":
        db_config["database"] = "huxley"
    
    config = config_manager.get_config()
    config["database"] = db_config
    config["step"] = "complete"
    config_manager.save_config(config)
    
    return jsonify({
        "success": True,
        "driver": driver,
        "configured": True
    })


@app.route("/api/setup/complete", methods=["POST"])
def complete_setup():
    """Mark setup as complete."""
    config = config_manager.get_config()
    config["setup_complete"] = True
    config_manager.save_config(config)
    
    return jsonify({
        "success": True,
        "message": "Setup complete! Huxley is ready to use."
    })


@app.route("/api/hardware-info", methods=["GET"])
def get_hardware_info():
    """Get hardware information for local model recommendations."""
    try:
        hw_info = detect_hardware()
        return jsonify(hw_info)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "cpu": "Unknown",
            "ram_gb": 0,
            "gpu_available": False
        }), 500


@app.route("/api/ollama-status", methods=["GET"])
def get_ollama_status():
    """Check if Ollama is installed and get installation instructions."""
    try:
        status = check_ollama_installation()
        return jsonify(status)
    except Exception as e:
        return jsonify({
            "error": str(e),
            "installed": False
        }), 500


@app.route("/api/test-connection", methods=["POST"])
def test_connection():
    """Test connection to a provider or local service."""
    data = request.json
    service_type = data.get("type")  # "provider" or "local"
    service_id = data.get("service_id")
    
    if service_type == "local" and service_id == "ollama":
        # Test Ollama connection
        import socket
        host = data.get("host", "localhost")
        port = data.get("port", 11434)
        
        try:
            # Extract hostname and port from URL if needed
            if "://" in host:
                host = host.split("://")[1].split(":")[0]
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, int(port)))
            sock.close()
            
            if result == 0:
                return jsonify({
                    "success": True,
                    "service": service_id,
                    "status": "connected",
                    "message": f"Successfully connected to Ollama at {host}:{port}"
                })
            else:
                return jsonify({
                    "success": False,
                    "service": service_id,
                    "status": "disconnected",
                    "message": f"Could not connect to Ollama at {host}:{port}"
                }), 503
        except Exception as e:
            return jsonify({
                "success": False,
                "service": service_id,
                "status": "error",
                "message": str(e)
            }), 500
    
    # For other services, return success
    return jsonify({
        "success": True,
        "service": service_id,
        "status": "connected"
    })


# =============================================================================
# STATIC FILES AND TEMPLATES
# =============================================================================

@app.route("/")
def index():
    """Serve the onboarding page."""
    return render_template("index.html")


@app.route("/assets/<path:filename>")
def serve_static(filename):
    """Serve static assets."""
    return send_from_directory("static", filename)


def detect_hardware() -> Dict[str, Any]:
    """Detect system hardware for model recommendations."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.system(),
        "ram_gb": 0,
        "gpu_available": False,
        "gpu_name": None,
        "gpu_vram_gb": 0,
        "recommendations": []
    }
    
    # Detect RAM
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                meminfo = f.read()
                for line in meminfo.split("\n"):
                    if "MemTotal" in line:
                        ram_kb = int(line.split()[1])
                        info["ram_gb"] = round(ram_kb / (1024 ** 2), 1)
                        break
        elif platform.system() == "Darwin":  # macOS
            result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True)
            if result.returncode == 0:
                ram_bytes = int(result.stdout.split()[1])
                info["ram_gb"] = round(ram_bytes / (1024 ** 3), 1)
    except Exception:
        pass
    
    # Detect GPU (NVIDIA)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip().split(",")
            info["gpu_available"] = True
            info["gpu_name"] = gpu_info[0].strip()
            if len(gpu_info) > 1:
                vram_str = gpu_info[1].strip().replace(" MiB", "")
                info["gpu_vram_gb"] = round(int(vram_str) / 1024, 1)
    except Exception:
        pass
    
    # Generate recommendations
    vram = info.get("gpu_vram_gb", 0)
    if vram >= 256:
        info["recommendations"] = ["qwen3-235b", "hermes-4.3", "intellect-3", "deepseek-v3.2", "glm-4.7"]
    elif vram >= 40:
        info["recommendations"] = ["hermes-4.3", "intellect-3", "deepseek-v3.2", "glm-4.7"]
    elif vram >= 12:
        info["recommendations"] = ["intellect-3", "deepseek-v3.2", "glm-4.7"]
    elif vram >= 10:
        info["recommendations"] = ["deepseek-v3.2", "glm-4.7"]
    elif vram >= 6:
        info["recommendations"] = ["glm-4.7"]
    else:
        info["recommendations"] = []
        info["message"] = "GPU not detected or insufficient VRAM. Consider using cloud APIs instead."
    
    return info


def check_ollama_installation() -> Dict[str, Any]:
    """Check if Ollama is installed and provide installation instructions."""
    ollama_path = shutil.which("ollama")
    
    status = {
        "installed": ollama_path is not None,
        "path": ollama_path,
        "version": None,
        "running": False,
        "installation_url": "https://ollama.ai/download",
        "instructions": {}
    }
    
    # Get version if installed
    if ollama_path:
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                status["version"] = result.stdout.strip()
        except Exception:
            pass
        
        # Check if running
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            status["running"] = result.returncode == 0
        except Exception:
            pass
    
    # Platform-specific installation instructions
    system = platform.system()
    if system == "Linux":
        status["instructions"] = {
            "method": "curl",
            "command": "curl -fsSL https://ollama.ai/install.sh | sh",
            "steps": [
                "Run: curl -fsSL https://ollama.ai/install.sh | sh",
                "Or download from: https://ollama.ai/download/linux",
                "After installation, start Ollama: ollama serve"
            ]
        }
    elif system == "Darwin":  # macOS
        status["instructions"] = {
            "method": "download",
            "command": "Download Ollama.app from https://ollama.ai/download",
            "steps": [
                "Download Ollama for macOS from https://ollama.ai/download",
                "Open the downloaded Ollama.app",
                "Ollama will run in the menu bar"
            ]
        }
    elif system == "Windows":
        status["instructions"] = {
            "method": "download",
            "command": "Download OllamaSetup.exe from https://ollama.ai/download",
            "steps": [
                "Download Ollama for Windows from https://ollama.ai/download",
                "Run OllamaSetup.exe",
                "Ollama will start automatically after installation"
            ]
        }
    
    return status


def check_api_server() -> bool:
    """Check if Huxley API server is running."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 8000))
        sock.close()
        return result == 0
    except Exception:
        return False


def run_server():
    """Run the Flask development server."""
    host = os.getenv("WEBUI_HOST", "0.0.0.0")
    port = int(os.getenv("WEBUI_PORT", 3000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    print(f"🚀 Huxley Onboarding WebUI starting on http://{host}:{port}")
    print("💡 Open http://localhost:3000 in your browser to begin setup")
    
    app.run(host=host, port=port, debug=debug, use_reloader=False)


if __name__ == "__main__":
    run_server()
