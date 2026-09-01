/**
 * Settings page JavaScript — RAG App
 *
 * Handles loading configuration, form interactions, saving via POST /api/config,
 * and testing LLM backend connections.
 */

(function () {
    "use strict";

    // DOM elements
    const backendSelect = document.getElementById("llm-backend");
    const modelInput = document.getElementById("llm-model");
    const ollamaUrlInput = document.getElementById("ollama-base-url");
    const openaiKeyInput = document.getElementById("api-key-openai");
    const geminiKeyInput = document.getElementById("api-key-gemini");
    const ollamaUrlGroup = document.getElementById("ollama-url-group");
    const openaiKeyGroup = document.getElementById("openai-key-group");
    const geminiKeyGroup = document.getElementById("gemini-key-group");
    const testConnectionBtn = document.getElementById("test-connection-btn");
    const connectionStatus = document.getElementById("connection-status");
    const chunkSizeInput = document.getElementById("chunk-size");
    const chunkOverlapInput = document.getElementById("chunk-overlap");
    const topKInput = document.getElementById("top-k");
    const embeddingModelInput = document.getElementById("embedding-model");
    const temperatureSlider = document.getElementById("temperature");
    const temperatureValue = document.getElementById("temperature-value");
    const maxTokensInput = document.getElementById("max-tokens");
    const integrationCheckbox = document.getElementById("integration-mode");
    const saveBtn = document.getElementById("save-btn");
    const saveStatus = document.getElementById("save-status");

    /**
     * Show/hide API key fields based on selected backend.
     */
    function updateBackendFields() {
        const backend = backendSelect.value;
        ollamaUrlGroup.style.display = backend === "ollama" ? "block" : "none";
        openaiKeyGroup.style.display = backend === "openai" ? "block" : "none";
        geminiKeyGroup.style.display = backend === "gemini" ? "block" : "none";
    }

    /**
     * Load current configuration from the server.
     */
    async function loadConfig() {
        try {
            const response = await fetch("/api/config");
            if (!response.ok) throw new Error("Failed to load config");
            const config = await response.json();

            // Populate form fields
            if (config.llm_backend) backendSelect.value = config.llm_backend;
            if (config.llm_model) modelInput.value = config.llm_model;
            if (config.ollama_base_url) ollamaUrlInput.value = config.ollama_base_url;
            if (config.api_key_openai) openaiKeyInput.value = config.api_key_openai;
            if (config.api_key_gemini) geminiKeyInput.value = config.api_key_gemini;
            if (config.chunk_size !== undefined) chunkSizeInput.value = config.chunk_size;
            if (config.chunk_overlap !== undefined) chunkOverlapInput.value = config.chunk_overlap;
            if (config.top_k !== undefined) topKInput.value = config.top_k;
            if (config.embedding_model) embeddingModelInput.value = config.embedding_model;
            if (config.temperature !== undefined) {
                temperatureSlider.value = config.temperature;
                temperatureValue.textContent = config.temperature;
            }
            if (config.max_tokens !== undefined) maxTokensInput.value = config.max_tokens;
            if (config.integration_mode !== undefined) integrationCheckbox.checked = config.integration_mode;

            updateBackendFields();
        } catch (err) {
            console.error("Error loading config:", err);
        }
    }

    /**
     * Save configuration to the server via POST /api/config.
     */
    async function saveConfig() {
        saveBtn.disabled = true;
        saveStatus.textContent = "Saving...";
        saveStatus.className = "save-status";

        const payload = {
            llm_backend: backendSelect.value,
            llm_model: modelInput.value,
            ollama_base_url: ollamaUrlInput.value,
            api_key_openai: openaiKeyInput.value,
            api_key_gemini: geminiKeyInput.value,
            chunk_size: parseInt(chunkSizeInput.value, 10),
            chunk_overlap: parseInt(chunkOverlapInput.value, 10),
            top_k: parseInt(topKInput.value, 10),
            embedding_model: embeddingModelInput.value,
            temperature: parseFloat(temperatureSlider.value),
            max_tokens: parseInt(maxTokensInput.value, 10),
            integration_mode: integrationCheckbox.checked,
        };

        try {
            const response = await fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const result = await response.json();

            if (result.status === "ok") {
                saveStatus.textContent = "✓ Configuration saved";
                saveStatus.className = "save-status success";
            } else {
                saveStatus.textContent = "✗ " + (result.message || "Save failed");
                saveStatus.className = "save-status error";
            }
        } catch (err) {
            saveStatus.textContent = "✗ Network error";
            saveStatus.className = "save-status error";
        } finally {
            saveBtn.disabled = false;
            // Clear status after 4 seconds
            setTimeout(() => {
                saveStatus.textContent = "";
                saveStatus.className = "save-status";
            }, 4000);
        }
    }

    /**
     * Test the LLM backend connection.
     */
    async function testConnection() {
        testConnectionBtn.disabled = true;
        setConnectionStatus("testing", "Testing...");

        const backend = backendSelect.value;
        const payload = {
            backend: backend,
            model: modelInput.value,
        };

        // Include relevant API key
        if (backend === "openai" && openaiKeyInput.value) {
            payload.api_key = openaiKeyInput.value;
        } else if (backend === "gemini" && geminiKeyInput.value) {
            payload.api_key = geminiKeyInput.value;
        }

        if (backend === "ollama" && ollamaUrlInput.value) {
            payload.base_url = ollamaUrlInput.value;
        }

        try {
            const response = await fetch("/api/config/test-connection", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            const result = await response.json();

            if (result.status === "connected") {
                setConnectionStatus("green", "Connected — " + (result.message || result.model));
            } else {
                setConnectionStatus("red", "Error — " + (result.message || "Connection failed"));
            }
        } catch (err) {
            setConnectionStatus("red", "Network error — could not reach server");
        } finally {
            testConnectionBtn.disabled = false;
        }
    }

    /**
     * Update the connection status indicator.
     * @param {string} color - "green", "gray", "red", or "testing"
     * @param {string} text - Status message
     */
    function setConnectionStatus(color, text) {
        const dot = connectionStatus.querySelector(".status-dot");
        const label = connectionStatus.querySelector(".status-text");
        dot.className = "status-dot " + color;
        label.textContent = text;
    }

    // Event listeners
    backendSelect.addEventListener("change", updateBackendFields);

    temperatureSlider.addEventListener("input", function () {
        temperatureValue.textContent = this.value;
    });

    testConnectionBtn.addEventListener("click", testConnection);
    saveBtn.addEventListener("click", saveConfig);

    // Initialize
    loadConfig();
})();
