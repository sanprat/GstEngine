# Tiny GST Engine Cloudflare Portal (`app.pybankers.com`)

100% Free Edge Portal for distributing Android APKs (`tinygstengine_v10.apk`) and on-device AI model weights (`model_int8.onnx`), complete with mobile QR code scanning and a passcode-protected Admin Upload Portal.

---

## ⚡ Architecture

- **Frontend**: Cloudflare Pages (Unlimited bandwidth, 300+ edge data centers).
- **Backend Storage**: Cloudflare R2 (10 GB free, **$0 egress / data transfer fees** forever).
- **Domain**: `app.pybankers.com` (with automatic free SSL).
- **Security**: Passcode-protected `/admin` upload portal (`X-Admin-Passcode`).

---

## 🚀 2-Minute Setup in Cloudflare Dashboard (Zero Coding Required)

### Step 1: Create the Free R2 Bucket
1. Open [Cloudflare Dashboard](https://dash.cloudflare.com/) ➔ Click **R2 Object Storage** in the sidebar.
2. Click **Create Bucket**.
3. Name it: **`gst-scanner-assets`** and click **Create**.
4. *(Optional initial upload)*: Drag and drop `tinygstengine_v10.apk` and `model_int8.onnx` into the bucket.

---

### Step 2: Deploy Cloudflare Pages
1. In Cloudflare Dashboard, click **Compute (Workers & Pages)** ➔ **Create Application** ➔ **Pages**.
2. Choose **Upload Assets**:
   - Project Name: `tinygst-portal`
   - Upload the `cloudflare_portal` folder (or drag and drop its files).
3. Click **Deploy Site**.

---

### Step 3: Bind the R2 Bucket & Admin Passcode
1. In your newly created Pages project, go to **Settings** ➔ **Functions**.
2. Scroll to **R2 bucket bindings**:
   - Variable name: `MY_BUCKET`
   - R2 bucket: Select `gst-scanner-assets`
3. Scroll to **Environment variables**:
   - Variable name: `ADMIN_PASSCODE`
   - Value: `pybankers2026` (or any custom secret passcode you want)
4. Click **Save and Deploy**.

---

### Step 4: Connect Domain `app.pybankers.com`
1. Go to **Custom domains** tab in your Pages project.
2. Click **Set up a custom domain**.
3. Enter: **`app.pybankers.com`**
4. Cloudflare will automatically add the CNAME DNS record and provision an SSL certificate in ~60 seconds.

---

## 💻 Alternative: Deploy in 10 Seconds via CLI

If you prefer deploying from your terminal on Mac:

```bash
cd cloudflare_portal

# 1. Log in to Cloudflare (opens browser)
npx wrangler login

# 2. Create the R2 bucket
npx wrangler r2 bucket create gst-scanner-assets

# 3. Deploy Pages with R2 binding
npx wrangler pages deploy public --project-name=tinygst-portal
```

---

## 📱 How to Use

### For Users / Friends:
- Visit `https://app.pybankers.com`
- Tap **Download APK (v10)** (downloads in 2–3 seconds at 30+ MB/s).
- Or point phone camera at the on-screen **QR Code** to install instantly.

### For Admin:
- Tap **🔒 Admin Portal** in the top navigation.
- Enter passcode: `pybankers2026`
- Drag and drop any new `.apk` or `.onnx` model file to update edge storage in real-time.
