"""Keep every test off the network and out of the user's Hugging Face cache.

The HF home lives in its own temporary directory, not under the test's
`tmp_path`: end-to-end tests make `tmp_path` the project root, and the effects
guard blocks writes inside the project tree, which would block the remote-code
modules cache too.
"""

from __future__ import annotations

import socket

import pytest


@pytest.fixture(autouse=True)
def hf_environment(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("hf_home")
    monkeypatch.setenv("HF_HOME", str(home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    for name in ("HF_HUB_CACHE", "HF_MODULES_CACHE", "TRANSFORMERS_CACHE"):
        monkeypatch.delenv(name, raising=False)

    def refuse(self, address):
        raise OSError(f"network disabled in tests (tried {address!r})")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    return home
