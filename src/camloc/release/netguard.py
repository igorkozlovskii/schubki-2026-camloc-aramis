"""In-process network guard (defence in depth; the gate proof is a network-less namespace). Blocks IP sockets and name resolution."""
from __future__ import annotations

import socket

_ACTIVE = False


class NetworkBlocked(RuntimeError):
    pass


def block_network() -> None:
    global _ACTIVE
    if _ACTIVE:
        return
    orig_connect, orig_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def _is_ip(sock) -> bool:
        return sock.family in (socket.AF_INET, socket.AF_INET6)

    def connect(self, addr):
        if _is_ip(self):
            raise NetworkBlocked(f"network access blocked by the release runner: connect {addr!r}")
        return orig_connect(self, addr)

    def connect_ex(self, addr):
        if _is_ip(self):
            raise NetworkBlocked(f"network access blocked by the release runner: connect_ex {addr!r}")
        return orig_connect_ex(self, addr)

    def getaddrinfo(*a, **k):
        raise NetworkBlocked(f"name resolution blocked by the release runner: {a[:2]!r}")

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, getaddrinfo
    _ACTIVE = True


def active() -> bool:
    return _ACTIVE
