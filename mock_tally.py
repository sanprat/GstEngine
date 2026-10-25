"""
Mock Tally server — simulates TallyPrime's :9000 HTTP import endpoint for testing.
Phase 0 of gst-engine.

Behavior (matches real Tally's documented responses):
  POST /  (Import Data envelope)
    -> parses TALLYMESSAGE blocks
    -> LEDGER ACTION=Create: returns CREATED (or ALTERED if name exists)
    -> VOUCHER ACTION=Create:
         REMOTEID already seen -> IGNORED (idempotency, real Tally behavior)
         ledger referenced but not yet created -> ERROR "...does not exist" (then we
             simulate real Tally: it does NOT auto-create; our client handles retry)
         ok -> CREATED
    Response: <ENVELOPE> with LINEERROR / success per message.

Run:  python mock_tally.py [port]   (default 9000)
"""

from __future__ import annotations
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {"ledgers": set(), "voucher_remoteids": set()}


def _messages(xml: str):
    # Split out each TALLYMESSAGE block
    for m in re.finditer(r"<TALLYMESSAGE[^>]*>(.*?)</TALLYMESSAGE>", xml, re.S):
        yield m.group(1)


def _tag(block: str, tagname: str):
    m = re.search(rf"<{tagname}>(.*?)</{tagname}>", block, re.S)
    return m.group(1).strip() if m else None


def _attr(block: str, tagname: str, attr: str):
    m = re.search(rf"<{tagname}\b([^>]*)>", block)
    if not m:
        return None
    a = re.search(rf'{attr}="([^"]*)"', m.group(1))
    return a.group(1) if a else None


def process_import(body: str) -> str:
    responses = []
    for block in _messages(body):
        if "<LEDGER " in block or "<LEDGER>" in block:
            name = _attr(block, "LEDGER", "NAME") or _tag(block, "NAME") or ""
            if name in STATE["ledgers"]:
                responses.append(f"<ALTERED>Altered: Ledger '{name}' already exists</ALTERED>")
            else:
                STATE["ledgers"].add(name)
                responses.append(f"<CREATED>Created: Ledger '{name}'</CREATED>")
        elif "<VOUCHER " in block:
            remote_id = _attr(block, "VOUCHER", "REMOTEID") or ""
            vtype = _attr(block, "VOUCHER", "VCHTYPE") or "Voucher"
            if remote_id and remote_id in STATE["voucher_remoteids"]:
                responses.append(f"<IGNORED>Ignored: Voucher REMOTEID '{remote_id}' already imported</IGNORED>")
                continue
            # Check ledger references
            ledgers = re.findall(r"<LEDGERNAME>(.*?)</LEDGERNAME>", block)
            missing = [l for l in ledgers if l not in STATE["ledgers"]]
            if missing:
                responses.append(
                    f"<LINEERROR>Error: Ledger '{missing[0]}' does not exist</LINEERROR>")
                continue
            STATE["voucher_remoteids"].add(remote_id)
            responses.append(f"<CREATED>Created: {vtype} '{remote_id}'</CREATED>")
        else:
            responses.append("<LINEERROR>Error: Unrecognized TALLYMESSAGE</LINEERROR>")
    return ("<ENVELOPE><BODY><IMPORTDATA><IMPORTRESULT>"
            + "".join(responses) +
            "</IMPORTRESULT></IMPORTDATA></BODY></ENVELOPE>")


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8", "replace")
        resp = process_import(body)
        data = resp.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/xml")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass  # quiet


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9000
    print(f"[mock-tally] listening on 0.0.0.0:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
