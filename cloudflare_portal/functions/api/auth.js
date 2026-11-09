/**
 * POST /api/auth
 * Verifies admin passcode
 */
export async function onRequestPost(context) {
  const { request, env } = context;
  try {
    const body = await request.json();
    const provided = (body.passcode || "").trim();
    const configured = (env.ADMIN_PASSCODE || "pybankers2026").trim();

    if (provided && provided === configured) {
      return new Response(JSON.stringify({ success: true, message: "Authenticated" }), {
        headers: { "Content-Type": "application/json" }
      });
    }

    return new Response(JSON.stringify({ success: false, error: "Invalid admin passcode" }), {
      status: 401,
      headers: { "Content-Type": "application/json" }
    });
  } catch (err) {
    return new Response(JSON.stringify({ success: false, error: err.message }), {
      status: 400,
      headers: { "Content-Type": "application/json" }
    });
  }
}
