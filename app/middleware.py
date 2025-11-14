import time
import json
import threading
from typing import Optional, Dict, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger=None):
        super().__init__(app)
        import logging
        self.logger = logger or logging.getLogger("app")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        ip = (request.client.host if request.client else None) or ""
        method = request.method
        path = request.url.path
        try:
            response: Response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise e
        finally:
            duration_ms = int((time.time() - start) * 1000)
            log: Dict[str, Any] = {
                "ts": int(time.time() * 1000),
                "ip": ip,
                "method": method,
                "path": path,
                "status": status,
                "duration_ms": duration_ms,
            }
            try:
                self.logger.info(json.dumps(log))
            except Exception:
                pass
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = int(max_requests)
        self.window = int(window_seconds)
        self.buckets: Dict[str, list] = {}
        self.lock = threading.Lock()

    async def dispatch(self, request: Request, call_next):
        ip = (request.client.host if request.client else "") or ""
        now = time.time()
        with self.lock:
            bucket = self.buckets.setdefault(ip, [])
            cutoff = now - self.window
            i = 0
            for i in range(len(bucket)):
                if bucket[i] >= cutoff:
                    break
            if i > 0:
                del bucket[:i]
            allowed = len(bucket) < self.max_requests
            if allowed:
                bucket.append(now)
        if not allowed:
            from starlette.responses import JSONResponse
            return JSONResponse({"error": "rate_limited"}, status_code=429)
        return await call_next(request)

