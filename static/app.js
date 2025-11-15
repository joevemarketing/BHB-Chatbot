// Top-level initializations
// app.js main script
const chatEl = document.getElementById("chat");
const inputEl = document.getElementById("message");
const sendBtn = document.getElementById("send");
const suggestionsEl = document.getElementById("suggestions");
const imageEl = document.getElementById("image");
const attachBtn = document.getElementById("attachBtn");
const newChatBtn = document.getElementById("newChat");
const ragToggleEl = document.getElementById("ragToggle");
const conciseToggleEl = document.getElementById("conciseToggle");
// KB upload controls
const kbFilesEl = document.getElementById("kbFiles");
const kbUploadBtn = document.getElementById("kbUpload");
const kbStatusEl = document.getElementById("kbStatus");

// Add a visible console log so we know the script loaded
console.log("app.js loaded ✓");

function showFatalError(msg) {
    try {
        const overlay = document.createElement("div");
        overlay.style.position = "fixed";
        overlay.style.top = "0";
        overlay.style.left = "0";
        overlay.style.right = "0";
        overlay.style.padding = "12px";
        overlay.style.background = "#ffefe8";
        overlay.style.color = "#b00020";
        overlay.style.fontFamily = "system-ui, sans-serif";
        overlay.style.zIndex = "99999";
        overlay.textContent = "Error: " + msg;
        document.body.appendChild(overlay);
    } catch (_) {}
}
window.addEventListener("error", (e) => {
    showFatalError(e.message || "Script error");
});
window.addEventListener("unhandledrejection", (e) => {
    const msg = (e?.reason?.message) ? e.reason.message : String(e.reason || "Unhandled rejection");
    showFatalError(msg);
});

// Safe session id generation (avoid ReferenceError if crypto is undefined)
const hasCrypto = typeof window !== "undefined" && typeof window.crypto !== "undefined" && typeof window.crypto.randomUUID === "function";
let sessionId = hasCrypto ? window.crypto.randomUUID() : String(Date.now());

// After addMessage()
function addMessage(role, content) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.textContent = content;
    chatEl && chatEl.appendChild(div);
    chatEl && (chatEl.scrollTop = chatEl.scrollHeight);
}

// Render assistant messages with safe HTML and autolinked URLs
function escapeHTML(str) {
    return str.replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
}
function autoLink(str) {
    const urlRegex = /(https?:\/\/[^\s)]+)/gm;
    return str.replace(
        urlRegex,
        (url) => `<a class="link-btn" href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer nofollow" referrerpolicy="no-referrer">Open</a>`
    );
}
function addAssistantMessage(content) {
    const div = document.createElement("div");
    div.className = "message assistant";
    const safe = autoLink(escapeHTML(content));
    div.innerHTML = safe.replace(/\n/g, "<br>");
    chatEl && chatEl.appendChild(div);
    chatEl && (chatEl.scrollTop = chatEl.scrollHeight);
}

// Suggestions per mode
const SUGGESTIONS = {
    smart_support: [
        "What’s the difference between front‑load and top‑load washers?",
        "Recommend a 55\" 4K TV under $600",
        "Best energy-efficient refrigerator"
    ],
    customer_support: [
        "What is your return policy?",
        "Do you offer extended warranty?",
        "How long does delivery take?"
    ],
    staff_training: [
        "How should I handle returns?",
        "Tips for upselling TVs",
        "Warranty process overview"
    ],
    shopping_assistant: [
        "Recommend a 55\" 4K TV under $600",
        "Best energy-efficient refrigerator",
        "Small appliance gift ideas"
    ]
};

// Render suggestions safely
function renderSuggestions() {
    if (!suggestionsEl) return; // guard if element is missing
    const items = SUGGESTIONS.smart_support || [];
    suggestionsEl.innerHTML = "";
    items.forEach(text => {
        const chip = document.createElement("button");
        chip.type = "button";
        chip.className = "chip";
        chip.textContent = text;
        chip.addEventListener("click", () => {
            if (!inputEl) return;
            inputEl.value = text;
            inputEl.focus();
        });
        suggestionsEl.appendChild(chip);
    });
}
renderSuggestions();

// Integrated image attach: trigger file picker and auto-run visual search
attachBtn?.addEventListener("click", () => {
    imageEl?.click();
});

imageEl?.addEventListener("change", async () => {
    const file = imageEl?.files?.[0];
    if (!file) return;
    // Show the image as part of the user message
    const wrapper = document.createElement("div");
    wrapper.className = "message user";
    const line = document.createElement("div");
    line.textContent = "[Image attached]";
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);
    img.alt = "Attached image";
    img.style.maxWidth = "220px";
    img.style.borderRadius = "10px";
    img.style.marginTop = "6px";
    wrapper.appendChild(line);
    wrapper.appendChild(img);
    chatEl && chatEl.appendChild(wrapper);
    chatEl && (chatEl.scrollTop = chatEl.scrollHeight);

    // Call visual search endpoint and render results like assistant
    attachBtn.disabled = true;
    attachBtn.classList.add("loading");
    try {
        const form = new FormData();
        form.append("file", file, file.name);
        const res = await fetch("/api/vision-search", { method: "POST", body: form });
        const data = await res.json();
        if (data.error) {
            addAssistantMessage("Visual search error: " + data.error);
            return;
        }
        const details = data.details || {};
        const results = data.results || [];
        const parts = [];
        if (details.brand) parts.push(`Brand: ${details.brand}`);
        if (details.model) parts.push(`Model: ${details.model}`);
        if (details.category) parts.push(`Category: ${details.category}`);
        if (details.keywords?.length) parts.push(`Keywords: ${details.keywords.slice(0, 6).join(", ")}`);
        const summary = "Detected details: " + (parts.join(" | ") || "(none)");
        addAssistantMessage(summary);
        if (results.length) {
            const container = document.createElement("div");
            const header = document.createElement("div");
            header.textContent = "Matching items:";
            container.appendChild(header);
            results.slice(0, 5).forEach(p => {
                const item = document.createElement("div");
                const currency = p.currency || (p._source === "bhb.com.my" ? "RM" : "$");
                const priceText = (p.price !== undefined && p.price !== null) ? ` — ${currency} ${p.price}` : "";
                const brandText = p.brand ? ` (${p.brand})` : "";
                if (p.permalink) {
                    const title = document.createElement("span");
                    title.textContent = `${p.name}`;
                    item.appendChild(title);
                    const meta = document.createElement("span");
                    meta.textContent = `${brandText}${priceText}`;
                    meta.style.marginLeft = "6px";
                    item.appendChild(meta);
                    const link = document.createElement("a");
                    link.href = p.permalink;
                    link.target = "_blank";
                    link.rel = "noopener noreferrer";
                    link.className = "link-btn";
                    link.textContent = "Open";
                    link.style.marginLeft = "8px";
                    item.appendChild(link);
                } else {
                    item.textContent = `${p.name}${brandText}${priceText}`;
                }
                container.appendChild(item);
            });
            const wrap = document.createElement("div");
            wrap.className = "message assistant";
            wrap.appendChild(container);
            chatEl && chatEl.appendChild(wrap);
            chatEl && (chatEl.scrollTop = chatEl.scrollHeight);
        } else {
            addAssistantMessage("No exact matches found. Try another angle or clearer photo.");
        }
    } catch (err) {
        addAssistantMessage("Network error during visual search. Please try again.");
    } finally {
        attachBtn.disabled = false;
        attachBtn.classList.remove("loading");
        try { imageEl.value = ""; } catch (_) {}
    }
});

// Send message
// function sendMessage()
async function sendMessage() {
    const msg = inputEl?.value?.trim();
    if (!msg) return;
    addMessage("user", msg);
    inputEl.value = "";
    sendBtn.disabled = true;
    sendBtn.classList.add("loading");
    // Show typing indicator for assistant
    const typingEl = document.createElement("div");
    typingEl.className = "message assistant typing";
    typingEl.textContent = "Assistant is typing…";
    chatEl && chatEl.appendChild(typingEl);
    chatEl && (chatEl.scrollTop = chatEl.scrollHeight);
    try {
        const isRag = !!(ragToggleEl && ragToggleEl.checked);
        const body = isRag ? {
            session_id: sessionId,
            message: msg,
            top_k: 5,
            concise: !!(conciseToggleEl && conciseToggleEl.checked)
        } : {
            session_id: sessionId,
            domain: "smart_support",
            message: msg,
            concise: !!(conciseToggleEl && conciseToggleEl.checked)
        };
        const url = isRag ? "/api/rag-chat" : "/api/chat";
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const data = await res.json();
        // Remove typing indicator before rendering final answer
        try { typingEl && typingEl.remove && typingEl.remove(); } catch (_) {}
        if (data.reply) {
            // Show the assistant reply text first
            addAssistantMessage(data.reply);
            // If RAG sources were returned, render citations
            if (Array.isArray(data.sources) && data.sources.length) {
                const container = document.createElement("div");
                const header = document.createElement("div");
                header.textContent = "Sources:";
                container.appendChild(header);
                data.sources.slice(0, 5).forEach(s => {
                    const row = document.createElement("div");
                    const title = s.title || "KB";
                    const path = s.path || "";
                    if (path) {
                        const link = document.createElement("a");
                        link.href = path.startsWith("/") ? path : ("/" + path);
                        link.target = "_blank";
                        link.rel = "noopener noreferrer";
                        link.className = "link-btn";
                        link.textContent = "Open";
                        const label = document.createElement("span");
                        label.textContent = ` ${title}`;
                        row.appendChild(label);
                        row.appendChild(link);
                    } else {
                        row.textContent = title;
                    }
                    container.appendChild(row);
                });
                const wrap = document.createElement("div");
                wrap.className = "message assistant";
                wrap.appendChild(container);
                chatEl && chatEl.appendChild(wrap);
                chatEl && (chatEl.scrollTop = chatEl.scrollHeight);
            }
            // If shopping payload is present, render product cards with Buy links
            if (Array.isArray(data.items) && data.items.length) {
                const container = document.createElement("div");
                container.className = "message assistant";
                data.items.slice(0, 5).forEach(item => {
                    const card = document.createElement("div");
                    card.className = "product-card";
                    // Preserve existing inline styles for now; add classes for consistent hooks
                    card.style.padding = "8px 10px";
                    card.style.margin = "6px 0";
                    card.style.border = "1px solid rgba(255,255,255,0.08)";
                    card.style.borderRadius = "10px";

                    const title = document.createElement("div");
                    title.className = "product-title";
                    const brandText = item.brand ? ` (${item.brand})` : "";
                    title.textContent = `${item.name}${brandText}`;

                    const meta = document.createElement("div");
                    meta.className = "product-meta";
                    const priceText = (typeof item.price === "number" && item.price > 0) ? `${item.currency || "RM"} ${item.price}` : "N/A";
                    meta.textContent = `Price: ${priceText}`;

                    const cta = document.createElement("a");
                    cta.className = "buy-link";
                    cta.textContent = "Buy at BHB";
                    cta.target = "_blank";
                    cta.rel = "noopener noreferrer";
                    cta.referrerPolicy = "no-referrer";
                    cta.href = item.link || "#";
                    cta.style.display = "inline-block";
                    cta.style.marginTop = "4px";
                    cta.style.background = "#f6c445";
                    cta.style.color = "#1a1405";
                    cta.style.fontWeight = "700";
                    cta.style.padding = "6px 10px";
                    cta.style.borderRadius = "8px";

                    // Optional product image if provided
                    if (item.image_url) {
                        const img = document.createElement("img");
                        img.className = "product-image";
                        img.src = item.image_url;
                        img.alt = item.name || "Product";
                        img.style.display = "block";
                        img.style.maxWidth = "120px";
                        img.style.borderRadius = "8px";
                        img.style.marginBottom = "6px";
                        card.appendChild(img);
                    }

                    card.appendChild(title);
                    card.appendChild(meta);
                    card.appendChild(cta);
                    container.appendChild(card);
                });
                chatEl && chatEl.appendChild(container);
                chatEl && (chatEl.scrollTop = chatEl.scrollHeight);
            }
        } else {
            try { typingEl && typingEl.remove && typingEl.remove(); } catch (_) {}
            addAssistantMessage("Error: " + (data.error || "unknown"));
        }
    } catch (err) {
        try { typingEl && typingEl.remove && typingEl.remove(); } catch (_) {}
        addAssistantMessage("Network error. Please try again.");
    } finally {
        sendBtn.disabled = false;
        sendBtn.classList.remove("loading");
    }
}

sendBtn?.addEventListener("click", sendMessage);
inputEl?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
});

// ---------- Knowledge Base Upload ----------
function setBtnLoading(btn, loading) {
    try {
        const spinner = btn?.querySelector?.(".btn-spinner");
        if (loading) {
            btn && btn.setAttribute("disabled", "disabled");
            if (spinner) spinner.style.display = "inline-block";
        } else {
            btn && btn.removeAttribute("disabled");
            if (spinner) spinner.style.display = "none";
        }
    } catch (_) {}
}

async function uploadKBFormData(fd) {
    if (!fd) throw new Error("FormData is required");
    const res = await fetch("/api/kb/upload", {
        method: "POST",
        body: fd,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data?.ok !== true) {
        throw new Error(data?.error || `Upload failed (${res.status})`);
    }
    return data;
}

async function uploadKB() {
    try {
        if (!kbFilesEl || !kbFilesEl.files || kbFilesEl.files.length === 0) {
            kbStatusEl && (kbStatusEl.textContent = "Choose file(s) to upload.");
            return;
        }
        setBtnLoading(kbUploadBtn, true);
        kbStatusEl && (kbStatusEl.textContent = "Uploading...");
        const fd = new FormData();
        Array.from(kbFilesEl.files).forEach((f) => fd.append("files", f));
        const data = await uploadKBFormData(fd);
        const titles = (data?.added || []).map((a) => a.title).filter(Boolean);
        kbStatusEl && (kbStatusEl.textContent = `Uploaded ${data?.count || titles.length} file(s): ${titles.join(", ")}`);
        try { kbFilesEl.value = ""; } catch (_) {}
    } catch (err) {
        kbStatusEl && (kbStatusEl.textContent = `Upload error: ${err?.message || String(err)}`);
    } finally {
        setBtnLoading(kbUploadBtn, false);
    }
}

kbUploadBtn?.addEventListener("click", uploadKB);

// Expose a test helper for automation to validate KB upload
window.uploadKBTest = async function () {
    const blob = new Blob(["Hello KB test content"], { type: "text/plain" });
    const file = new File([blob], "kb_test.txt", { type: "text/plain" });
    const fd = new FormData();
    fd.append("files", file);
    const data = await uploadKBFormData(fd);
    if (kbStatusEl) kbStatusEl.textContent = `Uploaded ${data?.count} file(s): ${(data?.added || []).map(a => a.title).join(', ')}`;
    return data;
};

// New Chat: start a fresh session and clear UI
function newChat() {
    try {
        sessionId = hasCrypto ? window.crypto.randomUUID() : String(Date.now());
        if (chatEl) chatEl.innerHTML = "";
        if (inputEl) { inputEl.value = ""; inputEl.focus(); }
        renderSuggestions();
        addAssistantMessage("Started a new chat. How can I help?");
    } catch (_) {}
}

newChatBtn?.addEventListener("click", newChat);