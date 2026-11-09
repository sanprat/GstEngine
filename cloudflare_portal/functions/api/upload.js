/**
 * POST /api/upload?filename=xyz
 * Saves incoming binary stream directly to Cloudflare R2 bucket
 */
export async function onRequestPost(context) {
  const { request, env } = context;

  // 1. Authenticate Admin Passcode
  const provided = (request.headers.get("X-Admin-Passcode") || "").trim();
  const configured = (env.ADMIN_PASSCODE || "pybankers2026").trim();

  if (!provided || provided !== configured) {
    return new Response(JSON.stringify({ error: "Unauthorized: Invalid admin passcode." }), {
      status: 401,
      headers: { "Content-Type": "application/json" }
    });
  }

  if (!env.MY_BUCKET) {
    return new Response(JSON.stringify({ error: "R2 Bucket binding (MY_BUCKET) is not configured." }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }

  // 2. Extract Filename
  const url = new URL(request.url);
  const filename = decodeURIComponent(url.searchParams.get("filename") || "").trim();

  if (!filename) {
    return new Response(JSON.stringify({ error: "Missing ?filename parameter." }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }

  // 3. Determine Content-Type
  let contentType = "application/octet-stream";
  if (filename.endsWith(".apk")) {
    contentType = "application/vnd.android.package-archive";
  } else if (filename.endsWith(".onnx")) {
    contentType = "application/octet-stream";
  }

  try {
    // 4. Stream directly to Cloudflare R2
    await env.MY_BUCKET.put(filename, request.body, {
      httpMetadata: { contentType }
    });

    return new Response(JSON.stringify({
      success: true,
      message: `File "${filename}" saved successfully to Cloudflare R2.`,
      url: `/api/download/${encodeURIComponent(filename)}`
    }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: `Upload error: ${err.message}` }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
