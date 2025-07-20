import asyncio
import time
from collections import deque
from typing import Deque, Tuple


class TokenRateLimiter:
    """a token-based rate limiter for managing api usage."""
    def __init__(self, tpm_limit: int, time_window_seconds: int = 60):
        self.tpm_limit = tpm_limit
        self.time_window_seconds = time_window_seconds
        self.token_usage: Deque[Tuple[float, int]] = deque()
        self.lock = asyncio.Lock()

    def _prune_old_requests(self) -> None:
        """removes token usage records that are outside the time window."""
        current_time = time.monotonic()
        while self.token_usage and current_time - self.token_usage[0][0] > self.time_window_seconds:
            self.token_usage.popleft()

    def _get_current_usage(self) -> int:
        """calculates the total token usage within the current time window."""
        self._prune_old_requests()
        return sum(tokens for _, tokens in self.token_usage)

    async def wait_for_capacity(self, tokens_needed: int) -> None:
        """
        waits until there is enough capacity in the rate limit to handle the request.
        """
        while True:
            async with self.lock:
                current_usage = self._get_current_usage()
                if current_usage + tokens_needed <= self.tpm_limit:
                    self.token_usage.append((time.monotonic(), tokens_needed))
                    return

            # if capacity is exceeded, calculate wait time
            time_to_wait = 0
            async with self.lock:
                if self.token_usage:
                    time_to_wait = self.time_window_seconds - (time.monotonic() - self.token_usage[0][0])
            
            await asyncio.sleep(max(1.0, time_to_wait / 2)) 