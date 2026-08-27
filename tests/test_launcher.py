from __future__ import annotations

import socket

from tools.find_free_port import find_free_port


def test_find_free_port_returns_bindable_port() -> None:
    port = find_free_port(18901, 18920)
    assert 18901 <= port <= 18920
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))
