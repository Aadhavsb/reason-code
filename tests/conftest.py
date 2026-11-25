"""Test-wide guard against accidental outbound transport."""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def block_outbound_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the hermetic suite honest; fake transports never need a real socket."""
    real_socket = socket.socket

    def is_loopback(address: object) -> bool:
        if not isinstance(address, tuple) or not address:
            return False
        return address[0] in {"127.0.0.1", "::1"}

    class NoOutboundSocket(real_socket):
        def connect(self, address: object) -> None:
            if not is_loopback(address):
                raise AssertionError(f"outbound socket blocked during tests: {address!r}")
            super().connect(address)

        def connect_ex(self, address: object) -> int:
            if not is_loopback(address):
                raise AssertionError(f"outbound socket blocked during tests: {address!r}")
            return super().connect_ex(address)

    monkeypatch.setattr(socket, "socket", NoOutboundSocket)
