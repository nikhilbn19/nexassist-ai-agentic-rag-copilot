import time
from collections import defaultdict
from threading import Lock
from fastapi import Request, HTTPException

class InMemoryRateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.rpm = requests_per_minute
        self.requests = defaultdict(list)
        self.lock = Lock()

    def check(self, request: Request) -> None:
        client_ip = request.client.host if request.client else "127.0.0.1"
        now = time.time()
        window = 60.0
        with self.lock:
            timestamps = self.requests[client_ip]
            self.requests[client_ip] = [t for t in timestamps if now - t < window]
            if len(self.requests[client_ip]) >= self.rpm:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please wait a moment before trying again."
                )
            self.requests[client_ip].append(now)

chat_limiter = InMemoryRateLimiter(requests_per_minute=30)
ingest_limiter = InMemoryRateLimiter(requests_per_minute=10)

