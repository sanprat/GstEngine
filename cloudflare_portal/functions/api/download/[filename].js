/**
 * GET /api/download/[filename]
 * Streams requested file directly from Cloudflare R2 bucket with high-speed edge caching
 */
export async function onRequestGet(context) {
  const { params, env, request } = context;
  const filename = decodeURIComponent(params.filename || "");

  if (!filename) {
    return new Response("Missing filename parameter", { status: 400 });
  }

  if (!env.MY_BUCKET) {
    return new Response("Cloudflare R2 Bucket binding (MY_BUCKET) is not configured.", { status: 500 });
  }

  // Get object from R2
  const object = await env.MY_BUCKET.get(filename, {
    onlyIf: request.headers
  });

  if (!object) {
    return new Response(`File "${filename}" not found in storage.`, { status: 404 });
  }

  const headers = new Headers();
  object.writeHttpMetadata(headers);
  headers.set("etag", object.httpEtag);
  headers.set("Cache-Control", "public, max-age=604800, immutable");

  // Determine appropriate content type & download disposition
  if (filename.endsWith(".apk")) {
    headers.set("Content-Type", "application/vnd.android.package-archive");
    headers.set("Content-Disposition", `attachment; filename="${filename}"`);
  } else if (filename.endsWith(".onnx")) {
    headers.set("Content-Type", "application/octet-stream");
    headers.set("Content-Disposition", `attachment; filename="${filename}"`);
  } else {
    headers.set("Content-Disposition", `attachment; filename="${filename}"`);
  }

  return new Response(object.body, {
    status: object.body ? 200 : 304,
    headers
  });
}
