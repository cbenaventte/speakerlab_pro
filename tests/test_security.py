import asyncio
import json
import unittest
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse

import api.index as api


def request_for(path="/api/simulate", headers=None, client=("127.0.0.1", 5000)):
    raw_headers = [
        (key.lower().encode(), str(value).encode())
        for key, value in (headers or {}).items()
    ]
    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": raw_headers,
        "client": client,
        "server": ("testserver", 80),
    })


class OperationalGuardTests(unittest.TestCase):
    def setUp(self):
        api._request_windows.clear()

    def test_rejects_oversized_declared_body(self):
        called = False

        async def next_handler(_request):
            nonlocal called
            called = True
            return JSONResponse({"ok": True})

        request = request_for(headers={"content-length": api.MAX_REQUEST_BYTES + 1})
        response = asyncio.run(api.operational_guards(request, next_handler))
        self.assertEqual(response.status_code, 413)
        self.assertFalse(called)

    def test_adds_security_headers(self):
        async def next_handler(_request):
            return JSONResponse({"ok": True})

        response = asyncio.run(
            api.operational_guards(request_for(path="/api/health"), next_handler)
        )
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertIn("frame-ancestors 'none'", response.headers["content-security-policy"])
        script_policy = response.headers["content-security-policy"].split("script-src", 1)[1].split(";", 1)[0]
        self.assertNotIn("unsafe-inline", script_policy)

    def test_rate_limit_returns_retry_after(self):
        async def next_handler(_request):
            return JSONResponse({"ok": True})

        api._request_windows["127.0.0.1"].extend(
            [api.time.monotonic()] * api.RATE_LIMIT_PER_MINUTE
        )
        response = asyncio.run(api.operational_guards(request_for(), next_handler))
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "60")

    def test_vercel_static_routes_have_security_headers(self):
        config = json.loads((Path(__file__).resolve().parents[1] / "vercel.json").read_text())
        for route in config["routes"]:
            headers = {key.lower(): value for key, value in route["headers"].items()}
            self.assertEqual(headers["x-content-type-options"], "nosniff")
            self.assertEqual(headers["x-frame-options"], "DENY")
            self.assertIn("frame-ancestors 'none'", headers["content-security-policy"])
            script_policy = headers["content-security-policy"].split("script-src", 1)[1].split(";", 1)[0]
            self.assertNotIn("unsafe-inline", script_policy)


if __name__ == "__main__":
    unittest.main()
