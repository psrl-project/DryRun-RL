"""Shared subprocess lifecycle management for profiling."""

from __future__ import annotations

import logging
import signal
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

dryrun_logger = logging.getLogger("dryrun")


class ProfileRunner:
    """
    Manages the lifecycle of a profiling subprocess (server or training run).

    Handles launch, health checking, and graceful shutdown.
    """

    def __init__(self, cmd: list[str], health_url: str | None = None, timeout: int = 300):
        self.cmd = cmd
        self.health_url = health_url
        self.timeout = timeout
        self.proc: subprocess.Popen | None = None

    def launch(self) -> subprocess.Popen:
        """Launch the subprocess."""
        dryrun_logger.info(f"Launching: {' '.join(self.cmd)}")
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return self.proc

    def wait_healthy(self) -> None:
        """Poll the health endpoint until the server is ready."""
        assert self.proc is not None, "Must call launch() first."
        assert self.health_url is not None, "No health URL configured."

        dryrun_logger.info(f"Waiting for server at {self.health_url} (timeout={self.timeout}s)...")
        start = time.monotonic()
        while time.monotonic() - start < self.timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"Server process exited with code {self.proc.returncode} "
                    f"before becoming healthy."
                )
            try:
                with urlopen(self.health_url, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.monotonic() - start
                        dryrun_logger.info(f"Server healthy after {elapsed:.1f}s.")
                        return
            except (URLError, OSError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError(
            f"Server did not become healthy within {self.timeout}s."
        )

    def shutdown(self, timeout: int = 30) -> None:
        """Gracefully shut down the subprocess."""
        if self.proc is None or self.proc.poll() is not None:
            return
        dryrun_logger.info("Shutting down profiling subprocess...")
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            dryrun_logger.warning("Subprocess did not exit cleanly, sending SIGKILL.")
            self.proc.kill()
            self.proc.wait(timeout=10)

    def read_output(self) -> str:
        """Read all stdout from the completed subprocess."""
        assert self.proc is not None, "Must call launch() first."
        if self.proc.stdout is not None:
            return self.proc.stdout.read()
        return ""

    def __enter__(self):
        self.launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return False
