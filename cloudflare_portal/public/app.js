/**
 * Tiny GST Engine Portal Client Logic
 * app.pybankers.com
 */

let selectedFile = null;
let currentAdminPasscode = sessionStorage.getItem("admin_passcode") || "";

document.addEventListener("DOMContentLoaded", () => {
  renderQrCode();
  setupDragAndDrop();
  if (currentAdminPasscode) {
    verifyStoredPasscode();
  }
});

// Tab Switcher
function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));

  if (tabName === "downloads") {
    document.getElementById("tab-downloads").classList.add("active");
    document.getElementById("section-downloads").classList.add("active");
  } else {
    document.getElementById("tab-admin").classList.add("active");
    document.getElementById("section-admin").classList.add("active");
    if (currentAdminPasscode) {
      loadBucketFiles();
    }
  }
}

// Copy Direct Link to Clipboard
function copyLink(path) {
  const fullUrl = window.location.origin + path;
  navigator.clipboard.writeText(fullUrl).then(() => {
    alert("Copied download link:\n" + fullUrl);
  }).catch(() => {
    prompt("Direct download link:", fullUrl);
  });
}

// Minimal Self-Contained QR Code Generator (Zero external dependencies)
function renderQrCode() {
  const canvas = document.getElementById("qr-canvas");
  if (!canvas) return;

  const url = window.location.origin + "/api/download/tinygstengine_v10.apk";
  
  // Use public QuickChart QR or offline fallback canvas renderer
  const ctx = canvas.getContext("2d");
  canvas.width = 160;
  canvas.height = 160;

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    ctx.drawImage(img, 0, 0, 160, 160);
  };
  img.onerror = () => {
    // Offline Canvas Fallback Pattern
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, 160, 160);
    ctx.fillStyle = "#0f172a";
    ctx.fillRect(10, 10, 40, 40);
    ctx.clearRect(18, 18, 24, 24);
    ctx.fillRect(24, 24, 12, 12);
    ctx.fillRect(110, 10, 40, 40);
    ctx.clearRect(118, 18, 24, 24);
    ctx.fillRect(124, 24, 12, 12);
    ctx.fillRect(10, 110, 40, 40);
    ctx.clearRect(18, 118, 24, 24);
    ctx.fillRect(24, 124, 12, 12);
    ctx.font = "bold 10px sans-serif";
    ctx.fillText("SCAN FOR APK", 42, 85);
  };
  // Uses Cloudflare edge friendly fast QR API
  img.src = `https://api.qrserver.com/v1/create-qr-code/?size=160x160&data=${encodeURIComponent(url)}&margin=1`;
}

// =================== ADMIN PORTAL LOGIC ===================

async function handleAdminLogin(e) {
  e.preventDefault();
  const passcode = document.getElementById("admin-passcode").value.trim();
  const errorEl = document.getElementById("auth-error");
  errorEl.textContent = "";

  try {
    const res = await fetch("/api/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passcode })
    });

    const data = await res.json();
    if (res.ok && data.success) {
      currentAdminPasscode = passcode;
      sessionStorage.setItem("admin_passcode", passcode);
      showAdminDashboard();
    } else {
      errorEl.textContent = data.error || "Invalid admin passcode.";
    }
  } catch (err) {
    errorEl.textContent = "Authentication error: " + err.message;
  }
}

async function verifyStoredPasscode() {
  try {
    const res = await fetch("/api/auth", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ passcode: currentAdminPasscode })
    });
    if (res.ok) {
      showAdminDashboard();
    } else {
      sessionStorage.removeItem("admin_passcode");
      currentAdminPasscode = "";
    }
  } catch (e) {
    // offline or error
  }
}

function showAdminDashboard() {
  document.getElementById("admin-auth-box").style.display = "none";
  document.getElementById("admin-dashboard").style.display = "flex";
  loadBucketFiles();
}

function handleAdminLogout() {
  sessionStorage.removeItem("admin_passcode");
  currentAdminPasscode = "";
  document.getElementById("admin-auth-box").style.display = "block";
  document.getElementById("admin-dashboard").style.display = "none";
  document.getElementById("admin-passcode").value = "";
}

// Drag & Drop Setup
function setupDragAndDrop() {
  const dropZone = document.getElementById("drop-zone");
  if (!dropZone) return;

  ["dragenter", "dragover"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(event => {
    dropZone.addEventListener(event, (e) => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
    }
  });
}

function handleFileSelect(e) {
  if (e.target.files && e.target.files[0]) {
    selectFile(e.target.files[0]);
  }
}

function selectFile(file) {
  selectedFile = file;
  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  document.getElementById("selected-file-info").textContent = `Selected: ${file.name} (${sizeMb} MB)`;
  document.getElementById("btn-upload").removeAttribute("disabled");

  // Auto-select radio option if name matches
  if (file.name.endsWith(".onnx")) {
    document.querySelector('input[name="target-name-type"][value="model_int8.onnx"]').checked = true;
    toggleCustomFileName(false);
  } else if (file.name.endsWith(".apk")) {
    document.querySelector('input[name="target-name-type"][value="tinygstengine_v10.apk"]').checked = true;
    toggleCustomFileName(false);
  }
}

function toggleCustomFileName(isCustom) {
  const customInput = document.getElementById("custom-file-name");
  customInput.style.display = isCustom ? "block" : "none";
  if (isCustom && selectedFile) {
    customInput.value = selectedFile.name;
  }
}

// Upload File via XHR to show live progress
function startUpload() {
  if (!selectedFile) return;

  let targetName = document.querySelector('input[name="target-name-type"]:checked').value;
  if (targetName === "custom") {
    targetName = document.getElementById("custom-file-name").value.trim();
    if (!targetName) {
      alert("Please enter a valid target filename.");
      return;
    }
  }

  const progressContainer = document.getElementById("upload-progress-container");
  const progressFill = document.getElementById("upload-progress-fill");
  const progressText = document.getElementById("upload-progress-text");
  const statusEl = document.getElementById("upload-status");
  const btnUpload = document.getElementById("btn-upload");

  progressContainer.style.display = "block";
  btnUpload.setAttribute("disabled", "true");
  statusEl.textContent = "Uploading to Cloudflare R2...";
  statusEl.style.color = "#94a3b8";

  const xhr = new XMLHttpRequest();
  xhr.open("POST", `/api/upload?filename=${encodeURIComponent(targetName)}`);
  xhr.setRequestHeader("X-Admin-Passcode", currentAdminPasscode);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const percent = Math.round((e.loaded / e.total) * 100);
      progressFill.style.width = percent + "%";
      const loadedMb = (e.loaded / (1024 * 1024)).toFixed(1);
      const totalMb = (e.total / (1024 * 1024)).toFixed(1);
      progressText.textContent = `${percent}% (${loadedMb} MB of ${totalMb} MB)`;
    }
  };

  xhr.onload = () => {
    btnUpload.removeAttribute("disabled");
    if (xhr.status === 200) {
      statusEl.textContent = `✅ Successfully uploaded "${targetName}" to Cloudflare R2!`;
      statusEl.style.color = "#34d399";
      progressFill.style.width = "100%";
      loadBucketFiles();
    } else {
      statusEl.textContent = `❌ Upload failed (${xhr.status}): ${xhr.responseText}`;
      statusEl.style.color = "#ef4444";
    }
  };

  xhr.onerror = () => {
    btnUpload.removeAttribute("disabled");
    statusEl.textContent = "❌ Network error during upload.";
    statusEl.style.color = "#ef4444";
  };

  xhr.send(selectedFile);
}

// Load Files List from R2
async function loadBucketFiles() {
  const container = document.getElementById("bucket-files-list");
  container.innerHTML = '<div class="loading">Fetching file inventory...</div>';

  try {
    const res = await fetch("/api/files", {
      headers: { "X-Admin-Passcode": currentAdminPasscode }
    });

    if (!res.ok) {
      container.innerHTML = `<div class="error-msg">Failed to load files: ${res.statusText}</div>`;
      return;
    }

    const data = await res.json();
    if (!data.files || data.files.length === 0) {
      container.innerHTML = '<div style="color: #94a3b8; padding: 1rem;">No files currently in bucket. Use the upload box above.</div>';
      return;
    }

    let html = `
      <table>
        <thead>
          <tr>
            <th>File Name</th>
            <th>Size</th>
            <th>Uploaded</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
    `;

    data.files.forEach(f => {
      const sizeMb = (f.size / (1024 * 1024)).toFixed(2);
      const dateStr = new Date(f.uploaded).toLocaleString();
      const downloadUrl = `/api/download/${encodeURIComponent(f.name)}`;

      html += `
        <tr>
          <td><strong>${f.name}</strong></td>
          <td>${sizeMb} MB</td>
          <td style="color: #94a3b8; font-size: 0.8rem;">${dateStr}</td>
          <td>
            <a href="${downloadUrl}" class="btn btn-secondary btn-sm" download style="display:inline-flex; margin-right: 0.4rem;">⬇️ Download</a>
            <button class="btn btn-danger btn-sm" onclick="deleteBucketFile('${f.name}')">🗑️ Delete</button>
          </td>
        </tr>
      `;
    });

    html += '</tbody></table>';
    container.innerHTML = html;
  } catch (err) {
    container.innerHTML = `<div class="error-msg">Error loading files: ${err.message}</div>`;
  }
}

// Delete File from R2
async function deleteBucketFile(filename) {
  if (!confirm(`Are you sure you want to delete "${filename}" from Cloudflare R2?`)) return;

  try {
    const res = await fetch(`/api/delete?filename=${encodeURIComponent(filename)}`, {
      method: "POST",
      headers: { "X-Admin-Passcode": currentAdminPasscode }
    });

    if (res.ok) {
      alert(`Deleted ${filename}`);
      loadBucketFiles();
    } else {
      const txt = await res.text();
      alert(`Delete failed: ${txt}`);
    }
  } catch (e) {
    alert("Delete error: " + e.message);
  }
}
