/**
 * GET /api/files
 * Lists all objects stored in the R2 bucket
 */
export async function onRequestGet(context) {
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

  try {
    const list = await env.MY_BUCKET.list({ limit: 100 });
    const files = list.objects.map(obj => ({
      name: obj.key,
      size: obj.size,
      uploaded: obj.uploaded
    }));

    return new Response(JSON.stringify({ files }), {
      headers: { "Content-Type": "application/json" }
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: err.message }), {
      status: 500,
      headers: { "Content-Type": "application/json" }
    });
  }
}
