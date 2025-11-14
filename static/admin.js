// Admin panel script: login + KB upload
const adminCodeEl = document.getElementById("adminCode");
const adminLoginBtn = document.getElementById("adminLogin");
const loginStatusEl = document.getElementById("loginStatus");
const kbPanelEl = document.getElementById("kbPanel");
const kbFilesEl = document.getElementById("kbFiles");
const kbUploadBtn = document.getElementById("kbUpload");
const kbStatusEl = document.getElementById("kbStatus");
const kbListPanelEl = document.getElementById("kbListPanel");
const kbRefreshBtn = document.getElementById("kbRefresh");
const kbListStatusEl = document.getElementById("kbListStatus");
const kbListEl = document.getElementById("kbList");

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

let ADMIN_TOKEN = null;

async function adminLogin() {
    try {
        const code = adminCodeEl?.value?.trim();
        if (!code) {
            loginStatusEl && (loginStatusEl.textContent = "Enter passcode.");
            return;
        }
        setBtnLoading(adminLoginBtn, true);
        loginStatusEl && (loginStatusEl.textContent = "Logging in...");
        const res = await fetch("/api/admin/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code })
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data?.ok !== true) {
            throw new Error(data?.error || `Login failed (${res.status})`);
        }
        // Capture token for header-based auth in webviews where cookies may not persist
        if (data?.token) {
            ADMIN_TOKEN = data.token;
        }
        loginStatusEl && (loginStatusEl.textContent = "Login successful.");
        // Show KB panel
        if (kbPanelEl) kbPanelEl.style.display = "block";
    } catch (err) {
        loginStatusEl && (loginStatusEl.textContent = `Error: ${err?.message || String(err)}`);
    } finally {
        setBtnLoading(adminLoginBtn, false);
    }
}

async function uploadKBFormData(fd) {
    if (!fd) throw new Error("FormData is required");
    const headers = {};
    if (ADMIN_TOKEN) headers["Authorization"] = `Bearer ${ADMIN_TOKEN}`;
    const res = await fetch("/api/kb/upload", {
        method: "POST",
        headers,
        body: fd,
        credentials: "include",
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

adminLoginBtn?.addEventListener("click", adminLogin);
kbUploadBtn?.addEventListener("click", uploadKB);

async function loadKBList() {
    try {
        kbListStatusEl && (kbListStatusEl.textContent = "Loading KB entries...");
        setBtnLoading(kbRefreshBtn, true);
        const headers = {};
        if (ADMIN_TOKEN) headers["Authorization"] = `Bearer ${ADMIN_TOKEN}`;
        const res = await fetch("/api/kb/list", { method: "GET", credentials: "include", headers });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data?.ok !== true) {
            throw new Error(data?.error || `List failed (${res.status})`);
        }
        renderKBList(data.items || []);
        kbListStatusEl && (kbListStatusEl.textContent = `Loaded ${data.count || (data.items || []).length} entries.`);
    } catch (err) {
        kbListStatusEl && (kbListStatusEl.textContent = `List error: ${err?.message || String(err)}`);
    } finally {
        setBtnLoading(kbRefreshBtn, false);
    }
}

function renderKBList(items) {
    if (!kbListEl) return;
    kbListEl.innerHTML = "";
    if (!items || items.length === 0) {
        const empty = document.createElement("div");
        empty.className = "kb-empty";
        empty.textContent = "No entries uploaded yet.";
        kbListEl.appendChild(empty);
        return;
    }
    items.slice(0, 200).forEach((it) => {
        const row = document.createElement("div");
        row.className = "kb-item";
        const title = document.createElement("div");
        title.className = "kb-item-title";
        title.textContent = it.title || "(untitled)";
        const meta = document.createElement("div");
        meta.className = "kb-item-meta";
        meta.textContent = `${(it.path || '').replace(/^\.\/?/, '')} — ${it.length || 0} chars`;
        const snippet = document.createElement("div");
        snippet.className = "kb-item-snippet";
        snippet.textContent = (it.snippet || '').slice(0, 240);
        row.appendChild(title);
        row.appendChild(meta);
        row.appendChild(snippet);
        kbListEl.appendChild(row);
    });
}

// After successful login, show list panel and load entries
async function afterLogin() {
    if (kbPanelEl) kbPanelEl.style.display = "block";
    if (kbListPanelEl) kbListPanelEl.style.display = "block";
    await loadKBList();
}

// Patch adminLogin to call afterLogin on success
const _adminLoginOrig = adminLogin;
adminLogin = async function() {
    try {
        await _adminLoginOrig();
        await afterLogin();
    } catch (_) {}
};

kbRefreshBtn?.addEventListener("click", loadKBList);