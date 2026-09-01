/**
 * RAG App — Vanilla JS frontend logic.
 *
 * Handles file upload (drag-and-drop + file picker),
 * Q&A chat via fetch POST /chat, and source chunk display.
 */

(function () {
    "use strict";

    // --- DOM Elements ---
    const uploadArea = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const uploadStatus = document.getElementById("upload-status");
    const documentList = document.getElementById("document-list");
    const modelSelect = document.getElementById("model-select");
    const embeddingSelect = document.getElementById("embedding-select");

    // --- Model selector: update config when changed ---
    if (modelSelect) {
        // Load current model from config
        fetch("/api/config")
            .then(r => r.json())
            .then(config => {
                if (config.llm_model) {
                    modelSelect.value = config.llm_model;
                }
                if (config.embedding_model && embeddingSelect) {
                    embeddingSelect.value = config.embedding_model;
                }
            })
            .catch(() => {});

        modelSelect.addEventListener("change", function () {
            fetch("/api/config", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ llm_model: modelSelect.value }),
            }).then(r => r.json())
              .then(data => console.log("Model updated:", data))
              .catch(err => console.warn("Failed to update model:", err));
        });
    }

    // --- Embedding model selector ---
    if (embeddingSelect) {
        embeddingSelect.addEventListener("change", function () {
            const newModel = embeddingSelect.value;
            if (confirm(
                "Changing the embedding model requires clearing the vector database " +
                "(existing embeddings have different dimensions).\n\n" +
                "Clear database and switch to " + newModel + "?"
            )) {
                fetch("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ embedding_model: newModel }),
                }).then(r => r.json())
                  .then(data => {
                      console.log("Embedding model updated:", data);
                      showUploadStatus("Embedding model changed to " + newModel + ". Please re-upload your documents.", "success");
                  })
                  .catch(err => console.warn("Failed to update embedding model:", err));
            } else {
                // Revert the dropdown
                fetch("/api/config")
                    .then(r => r.json())
                    .then(config => {
                        if (config.embedding_model) {
                            embeddingSelect.value = config.embedding_model;
                        }
                    });
            }
        });
    }
    const chatMessages = document.getElementById("chat-messages");
    const chatForm = document.getElementById("chat-form");
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const sourcesPanel = document.getElementById("sources-panel");
    const sourcesToggle = document.getElementById("sources-toggle");
    const sourcesContent = document.getElementById("sources-content");

    // Track uploaded documents
    const uploadedDocs = [];

    // --- File Upload Handling ---

    // Click to open file picker
    uploadArea.addEventListener("click", function () {
        fileInput.click();
    });

    // File selected via picker
    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
            fileInput.value = "";
        }
    });

    // Drag-and-drop events
    uploadArea.addEventListener("dragover", function (e) {
        e.preventDefault();
        uploadArea.classList.add("dragover");
    });

    uploadArea.addEventListener("dragleave", function (e) {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
    });

    uploadArea.addEventListener("drop", function (e) {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    /**
     * Upload a file to the server via POST /upload.
     */
    function handleFileUpload(file) {
        // Validate file type
        const allowedTypes = ["application/pdf", "text/plain", "text/markdown"];
        const allowedExtensions = [".pdf", ".txt", ".md"];
        const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();

        if (!allowedExtensions.includes(ext)) {
            showUploadStatus("Only PDF, TXT, and MD files are supported.", "error");
            return;
        }

        showUploadStatus("Uploading...", "loading");

        const formData = new FormData();
        formData.append("file", file);

        fetch("/upload", {
            method: "POST",
            body: formData,
        })
            .then(function (response) {
                return response.json().then(function (data) {
                    return { ok: response.ok, data: data };
                });
            })
            .then(function (result) {
                const data = result.data;
                if (data.status === "success" || data.status === "partial") {
                    showUploadStatus(data.message, "success");
                    addDocumentToList(data.filename || file.name);
                } else {
                    showUploadStatus(data.message || "Upload failed.", "error");
                }
            })
            .catch(function (err) {
                showUploadStatus("Upload failed: " + err.message, "error");
            });
    }

    /**
     * Display upload status message.
     */
    function showUploadStatus(message, type) {
        uploadStatus.innerHTML =
            '<span class="status-' + type + '">' + escapeHtml(message) + "</span>";
    }

    /**
     * Add a document to the uploaded documents list.
     */
    function addDocumentToList(filename) {
        if (uploadedDocs.includes(filename)) return;
        uploadedDocs.push(filename);
        refreshDocumentList();
    }

    /**
     * Re-render the document list.
     */
    function refreshDocumentList() {
        if (uploadedDocs.length === 0) {
            documentList.innerHTML =
                '<li class="empty-state">No documents uploaded yet</li>';
            return;
        }

        documentList.innerHTML = "";
        uploadedDocs.forEach(function (doc) {
            var li = document.createElement("li");
            var icon = doc.endsWith(".pdf") ? "📕" : "📄";
            li.innerHTML =
                '<span class="doc-icon">' + icon + "</span>" + escapeHtml(doc);
            documentList.appendChild(li);
        });
    }

    // --- Chat Handling ---

    chatForm.addEventListener("submit", function (e) {
        e.preventDefault();
        var query = chatInput.value.trim();
        if (!query) return;

        // Clear welcome message on first query
        var welcome = chatMessages.querySelector(".welcome-message");
        if (welcome) welcome.remove();

        // Add user message
        appendMessage(query, "user");
        chatInput.value = "";
        sendBtn.disabled = true;

        // Send query to backend
        fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query }),
        })
            .then(function (response) {
                return response.json().then(function (data) {
                    return { ok: response.ok, data: data };
                });
            })
            .then(function (result) {
                var data = result.data;
                if (data.error) {
                    appendMessage("Error: " + data.error, "error");
                } else {
                    appendMessage(data.answer, "assistant", data.model);
                    displaySources(data.sources || []);
                }
            })
            .catch(function (err) {
                appendMessage("Request failed: " + err.message, "error");
            })
            .finally(function () {
                sendBtn.disabled = false;
                chatInput.focus();
            });
    });

    /**
     * Append a message bubble to the chat.
     */
    function appendMessage(text, role, model) {
        var div = document.createElement("div");
        div.className = "message " + role;
        div.innerHTML = escapeHtml(text);

        if (model && role === "assistant") {
            var tag = document.createElement("span");
            tag.className = "model-tag";
            tag.textContent = "Model: " + model;
            div.appendChild(tag);
        }

        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // --- Source Chunks Display ---

    /**
     * Display source chunks below the chat.
     */
    function displaySources(sources) {
        if (!sources || sources.length === 0) {
            sourcesPanel.hidden = true;
            return;
        }

        sourcesPanel.hidden = false;
        sourcesContent.innerHTML = "";

        sources.forEach(function (source, idx) {
            var chunk = document.createElement("div");
            chunk.className = "source-chunk";

            var text = source.text || "";
            var score = source.score != null ? source.score.toFixed(3) : "N/A";
            var meta = source.metadata || {};
            var metaStr = meta.source || "unknown";

            chunk.innerHTML =
                "<p>" +
                escapeHtml(text.substring(0, 300)) +
                (text.length > 300 ? "..." : "") +
                "</p>" +
                '<div class="source-meta">' +
                "Source: " +
                escapeHtml(metaStr) +
                ' | Relevance: <span class="source-score">' +
                score +
                "</span>" +
                "</div>";

            sourcesContent.appendChild(chunk);
        });
    }

    // Source toggle
    sourcesToggle.addEventListener("click", function () {
        var isHidden = sourcesContent.hidden;
        sourcesContent.hidden = !isHidden;
        sourcesToggle.textContent = isHidden
            ? "▼ Hide Source Chunks"
            : "▶ Show Source Chunks";
    });

    // --- Utilities ---

    /**
     * Escape HTML to prevent XSS.
     */
    function escapeHtml(text) {
        var div = document.createElement("div");
        div.appendChild(document.createTextNode(text));
        return div.innerHTML;
    }

    // --- Clear Vector Collection ---
    var clearCollectionBtn = document.getElementById("clear-collection-btn");
    if (clearCollectionBtn) {
        clearCollectionBtn.addEventListener("click", function () {
            if (!confirm("Clear all stored document embeddings? You will need to re-upload documents.")) {
                return;
            }

            clearCollectionBtn.disabled = true;
            clearCollectionBtn.textContent = "⏳ Clearing...";

            fetch("/api/collection/clear", { method: "POST" })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data.status === "ok") {
                        showUploadStatus(data.message, "success");
                        // Reset document list
                        uploadedDocs.length = 0;
                        refreshDocumentList();
                    } else {
                        showUploadStatus("Error: " + (data.error || "Unknown error"), "error");
                    }
                })
                .catch(function (err) {
                    showUploadStatus("Failed to clear collection: " + err.message, "error");
                })
                .finally(function () {
                    clearCollectionBtn.disabled = false;
                    clearCollectionBtn.textContent = "🗑️ Clear Vector Collection";
                });
        });
    }
})();
