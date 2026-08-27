#!/usr/bin/env python3
"""Return the first free localhost TCP port in a small Streamlit-friendly range."""
from __future__ import annotations

import socket
import sys


def find_free_port(start: int = 8501, end: int = 8520) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free localhost port found in {start}-{end}")


if __name__ == "__main__":
    try:
        print(find_free_port())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
