/* ============================================================================
   Huxley Onboarding - JavaScript Application
   ============================================================================ */

// State management
let state = {
    currentStep: 'mode',
    mode: null,
    providers: {},
    selectedModel: null,
    database: {
        driver: 'sqlite'
    },
    hardwareInfo: null,
    ollamaStatus: null,
    localModels: []
};

const steps = ['mode', 'providers', 'local-setup', 'models', 'database', 'complete'];
const apiProviders = [];

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    await loadProviders();
    updateProgress();
});

// ============================================================================
// Step Navigation
// ============================================================================

async function selectMode(mode) {
    state.mode = mode;
    
    try {
        const response = await fetch('/api/setup/inference-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode })
        });
        
        if (!response.ok) throw new Error('Failed to save mode');
        
        if (mode === 'api') {
            nextStep('providers');
        } else {
            // Load hardware info and Ollama status before showing local setup
            await loadHardwareInfo();
            await checkOllamaStatus();
            await loadLocalModels();
            nextStep('local-setup');
        }
    } catch (error) {
        showMessage('Error saving configuration: ' + error.message, 'error');
    }
}

function nextStep(stepName) {
    const stepIndex = steps.indexOf(stepName);
    if (stepIndex === -1) return;
    
    state.currentStep = stepName;
    updateSteps();
    updateProgress();
}

function goBack() {
    const currentIndex = steps.indexOf(state.currentStep);
    if (currentIndex > 0) {
        const previousStep = steps[currentIndex - 1];
        state.currentStep = previousStep;
        updateSteps();
        updateProgress();
    }
}

function updateSteps() {
    // Hide all steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.add('hidden');
        step.classList.remove('active');
    });
    
    // Show current step
    const activeStep = document.getElementById(`step-${state.currentStep}`);
    if (activeStep) {
        activeStep.classList.remove('hidden');
        activeStep.classList.add('active');
    }
}

function updateProgress() {
    const currentIndex = steps.indexOf(state.currentStep);
    const progress = ((currentIndex + 1) / steps.length) * 100;
    document.getElementById('progress-fill').style.width = progress + '%';
}

// ============================================================================
// Providers
// ============================================================================

async function loadProviders() {
    try {
        const response = await fetch('/api/providers');
        if (!response.ok) throw new Error('Failed to load providers');
        
        const data = await response.json();
        renderProviders(data.api);
    } catch (error) {
        showMessage('Error loading providers: ' + error.message, 'error');
    }
}

function renderProviders(providers) {
    const container = document.getElementById('providers-list');
    container.innerHTML = '';
    
    providers.forEach(provider => {
        const card = document.createElement('div');
        card.className = 'provider-item';
        
        card.innerHTML = `
            <div class="provider-name">${provider.name}</div>
            <p class="provider-desc">${provider.description}</p>
            <div class="provider-form">
                <input type="password" 
                       placeholder="API Key" 
                       id="key-${provider.id}"
                       class="provider-input">
                <small>
                    <a href="${provider.url}" target="_blank">Get API key →</a>
                </small>
                <button class="btn btn-secondary" style="width: 100%; margin-top: 8px;"
                        onclick="saveProvider('${provider.id}', '${provider.name}')">
                    Save
                </button>
            </div>
        `;
        
        container.appendChild(card);
    });
}

async function saveProvider(providerId, providerName) {
    const apiKey = document.getElementById(`key-${providerId}`).value;
    
    if (!apiKey.trim()) {
        showMessage('Please enter an API key', 'error');
        return;
    }
    
    try {
        const response = await fetch('/api/setup/configure-provider', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider_id: providerId,
                api_key: apiKey
            })
        });
        
        if (!response.ok) throw new Error('Failed to save provider');
        
        state.providers[providerId] = apiKey;
        showMessage(`${providerName} configured successfully`, 'success');
        
        // Load models for this provider
        loadModels(providerId);
    } catch (error) {
        showMessage('Error: ' + error.message, 'error');
    }
}

// ============================================================================
// Models
// ============================================================================

async function loadModels(providerId) {
    // This would load available models from the provider
    // For now, show some defaults
    const models = {
        openai: ['gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'],
        anthropic: ['claude-opus', 'claude-sonnet', 'claude-haiku'],
        google: ['gemini-pro', 'gemini-pro-vision'],
        cohere: ['command-a', 'command-r'],
        openrouter: ['gpt-5.2', 'claude-4.5-sonnet', 'intellect-3']
    };
    
    const container = document.getElementById('models-list');
    container.innerHTML = '';
    
    (models[providerId] || []).forEach(model => {
        const option = document.createElement('div');
        option.className = 'model-option';
        
        option.innerHTML = `
            <input type="radio" name="selected-model" value="${model}">
            <label>${model}</label>
        `;
        
        container.appendChild(option);
    });
}

async function selectModel() {
    const selected = document.querySelector('input[name="selected-model"]:checked');
    if (!selected) {
        showMessage('Please select a model', 'error');
        return;
    }
    
    state.selectedModel = selected.value;
    
    try {
        const response = await fetch('/api/setup/select-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                provider: Object.keys(state.providers)[0],
                model: state.selectedModel
            })
        });
        
        if (!response.ok) throw new Error('Failed to save model selection');
        
        nextStep('database');
    } catch (error) {
        showMessage('Error: ' + error.message, 'error');
    }
}

// ============================================================================
// Database Configuration
// ============================================================================

function updateDbForm() {
    const driver = document.querySelector('input[name="db-driver"]:checked').value;
    const formContainer = document.getElementById('db-form');
    
    if (driver === 'sqlite') {
        formContainer.innerHTML = `
            <div class="form-group">
                <label>Database Path</label>
                <input type="text" value="~/.huxley/huxley.db" disabled>
            </div>
            <p class="subtitle">SQLite will be stored locally in your Huxley data directory.</p>
        `;
    } else {
        formContainer.innerHTML = `
            <div class="form-group">
                <label>Host</label>
                <input type="text" id="db-host" value="postgres">
            </div>
            <div class="form-group">
                <label>Port</label>
                <input type="text" id="db-port" value="5432">
            </div>
            <div class="form-group">
                <label>Database Name</label>
                <input type="text" id="db-name" value="huxley">
            </div>
            <div class="form-group">
                <label>Username</label>
                <input type="text" id="db-user" value="huxley">
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" id="db-password">
            </div>
        `;
    }
    
    state.database.driver = driver;
}

async function completeDatabaseSetup() {
    const driver = document.querySelector('input[name="db-driver"]:checked').value;
    
    let dbConfig = { driver };
    
    if (driver === 'postgresql') {
        dbConfig = {
            driver,
            host: document.getElementById('db-host')?.value,
            port: parseInt(document.getElementById('db-port')?.value || 5432),
            database: document.getElementById('db-name')?.value,
            username: document.getElementById('db-user')?.value,
            password: document.getElementById('db-password')?.value
        };
    }
    
    try {
        const response = await fetch('/api/setup/database', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(dbConfig)
        });
        
        if (!response.ok) throw new Error('Failed to save database configuration');
        
        completeSetup();
    } catch (error) {
        showMessage('Error: ' + error.message, 'error');
    }
}

// ============================================================================
// Local Setup (Ollama)
// ============================================================================

async function loadHardwareInfo() {
    try {
        const response = await fetch('/api/hardware-info');
        if (!response.ok) throw new Error('Failed to load hardware info');
        
        state.hardwareInfo = await response.json();
        displayHardwareInfo(state.hardwareInfo);
    } catch (error) {
        showMessage('Error detecting hardware: ' + error.message, 'error');
        displayHardwareInfo({ error: error.message });
    }
}

function displayHardwareInfo(info) {
    document.getElementById('hw-cpu').textContent = info.cpu || 'Unknown';
    document.getElementById('hw-ram').textContent = info.ram_gb ? `${info.ram_gb} GB` : 'Unknown';
    document.getElementById('hw-gpu').textContent = info.gpu_available ? 
        (info.gpu_name || 'Yes') : 'Not detected';
    document.getElementById('hw-vram').textContent = info.gpu_vram_gb ? 
        `${info.gpu_vram_gb} GB` : 'N/A';
    
    const recommendEl = document.getElementById('hw-recommendation');
    if (info.recommendations && info.recommendations.length > 0) {
        recommendEl.innerHTML = `
            <strong>✅ Recommended models for your hardware:</strong><br>
            ${info.recommendations.join(', ')}
        `;
        recommendEl.className = 'hw-recommendation';
    } else if (info.message) {
        recommendEl.innerHTML = `<strong>⚠️ ${info.message}</strong>`;
        recommendEl.className = 'hw-recommendation warning';
    }
}

async function checkOllamaStatus() {
    try {
        const response = await fetch('/api/ollama-status');
        if (!response.ok) throw new Error('Failed to check Ollama status');
        
        state.ollamaStatus = await response.json();
        displayOllamaStatus(state.ollamaStatus);
    } catch (error) {
        showMessage('Error checking Ollama: ' + error.message, 'error');
        displayOllamaStatus({ error: error.message, installed: false });
    }
}

function displayOllamaStatus(status) {
    const checkEl = document.getElementById('ollama-check');
    const guideEl = document.getElementById('ollama-install-guide');
    
    if (status.installed) {
        checkEl.innerHTML = `
            <span class="status-indicator">✅</span>
            <div>
                <strong>Ollama is installed!</strong>
                ${status.version ? `<br><small>${status.version}</small>` : ''}
                ${status.running ? '<br><small style="color: var(--success);">✓ Service is running</small>' : 
                  '<br><small style="color: var(--warning);">⚠ Service may not be running</small>'}
            </div>
        `;
        checkEl.className = 'status-check installed';
        guideEl.classList.add('hidden');
    } else {
        checkEl.innerHTML = `
            <span class="status-indicator">⚠️</span>
            <div>
                <strong>Ollama not found</strong>
                <br><small>Installation required to use local models</small>
            </div>
        `;
        checkEl.className = 'status-check not-installed';
        
        // Show installation guide
        if (status.instructions) {
            guideEl.innerHTML = `
                <h4>📥 How to Install Ollama</h4>
                ${status.instructions.command ? 
                    `<div class="install-command">${status.instructions.command}</div>` : ''}
                <ol>
                    ${status.instructions.steps.map(step => `<li>${step}</li>`).join('')}
                </ol>
                <p style="margin-top: 12px;">
                    <a href="${status.installation_url}" target="_blank" 
                       style="color: var(--primary); text-decoration: none;">
                        📖 Official Installation Guide →
                    </a>
                </p>
            `;
            guideEl.classList.remove('hidden');
        }
    }
}

async function loadLocalModels() {
    try {
        const response = await fetch('/api/providers');
        if (!response.ok) throw new Error('Failed to load models');
        
        const data = await response.json();
        if (data.local && data.local[0] && data.local[0].models) {
            state.localModels = data.local[0].models;
            displayLocalModels(state.localModels);
        }
    } catch (error) {
        showMessage('Error loading models: ' + error.message, 'error');
    }
}

function displayLocalModels(models) {
    const container = document.getElementById('ollama-models');
    container.innerHTML = '';
    
    const vram = state.hardwareInfo?.gpu_vram_gb || 0;
    const recommendations = state.hardwareInfo?.recommendations || [];
    
    models.forEach((model, index) => {
        const canRun = vram >= model.min_vram_gb;
        const efficiency = calculateEfficiency(vram, model);
        const isRecommended = recommendations.includes(model.id);
        
        const card = document.createElement('div');
        card.className = `model-card ${canRun ? '' : 'disabled'}`;
        card.onclick = canRun ? () => selectLocalModel(model.id) : null;
        
        card.innerHTML = `
            <div class="model-header">
                <div class="model-name">${model.name}</div>
                <div class="efficiency-badge efficiency-${efficiency.class}">
                    ${efficiency.label}
                </div>
            </div>
            <div class="model-huggingface">🤗 ${model.huggingface}</div>
            <div class="model-specs">
                <div class="model-spec">
                    <span>💾</span>
                    <span>${model.size_gb} GB</span>
                </div>
                <div class="model-spec">
                    <span>🎮</span>
                    <span>Min ${model.min_vram_gb} GB VRAM</span>
                </div>
                <div class="model-spec">
                    <span>⚡</span>
                    <span>Rec ${model.recommended_vram_gb} GB</span>
                </div>
            </div>
            <div class="model-requirement ${canRun ? 'requirement-met' : 'requirement-unmet'}">
                ${canRun ? 
                    (isRecommended ? '✅ Recommended for your hardware' : '✅ Can run on your hardware') : 
                    `❌ Requires ${model.min_vram_gb} GB VRAM (you have ${vram} GB)`}
            </div>
        `;
        
        container.appendChild(card);
    });
}

function calculateEfficiency(vram, model) {
    if (vram < model.min_vram_gb) {
        return { class: 'unavailable', label: 'Unavailable' };
    }
    
    const ratio = vram / model.recommended_vram_gb;
    
    if (ratio >= 1.5) {
        return { class: 'excellent', label: 'Excellent' };
    } else if (ratio >= 1.0) {
        return { class: 'good', label: 'Good' };
    } else if (ratio >= 0.8) {
        return { class: 'moderate', label: 'Moderate' };
    } else {
        return { class: 'poor', label: 'Poor' };
    }
}

function selectLocalModel(modelId) {
    // Remove selected class from all cards
    document.querySelectorAll('.model-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Add selected class to clicked card
    event.currentTarget.classList.add('selected');
    
    state.selectedModel = modelId;
}

async function testOllamaConnection() {
    const host = document.getElementById('ollama-host').value;
    
    try {
        const response = await fetch('/api/test-connection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                type: 'local',
                service_id: 'ollama',
                host
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            showMessage('✓ Connected to Ollama successfully!', 'success');
        } else {
            showMessage('✗ ' + (data.message || 'Could not connect to Ollama'), 'error');
        }
    } catch (error) {
        showMessage('Connection error: ' + error.message, 'error');
    }
}

// ============================================================================
// Completion
// ============================================================================

async function completeSetup() {
    try {
        const response = await fetch('/api/setup/complete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) throw new Error('Failed to complete setup');
        
        nextStep('complete');
    } catch (error) {
        showMessage('Error: ' + error.message, 'error');
    }
}

function closeOnboarding() {
    // Redirect to API documentation or dashboard
    window.location.href = 'http://localhost:8000/docs';
}

// ============================================================================
// UI Helpers
// ============================================================================

function showMessage(text, type = 'success') {
    const messageEl = document.getElementById('status-message');
    messageEl.textContent = text;
    messageEl.className = `status-message ${type === 'error' ? 'error' : ''}`;
    
    setTimeout(() => {
        messageEl.classList.add('hidden');
    }, 4000);
}
