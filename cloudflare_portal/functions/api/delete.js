/**
 * POST /api/delete?filename=xyz
 * Deletes an object from the R2 bucket
 */
export async function onRequestPost(context) {
  const { request, env } = context;

  // Verify Admin Passcode
  const provided = (request.headers.get("X-Admin-Passcode") || "").trim();
  const configured = (env.ADMIN_PASSCODE || "pybankers2026").trim();

  if (!provided || provided !== configured) {
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
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

  const url = new URL(request.url);
  const filename = decodeURIComponent(url.searchParams.get("filename") || "").trim();

  if (!filename) {
    return new Response(JSON.stringify({ error: "Missing ?filename parameter." }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }

  try {
    await env.MY_BUCKET.delete(filename);
    return new Response(JSON.stringify({ success: true, message: `Deleted ${filename}` }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
