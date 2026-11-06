"""Optional tests for the official Mem0 adapter.

The adapter is skipped when the repository-local ``.vendor`` copy of ``mem0ai``
is not importable (for example on a fresh checkout that did not run the vendor
setup).  The test exercises the official package end to end with
``infer=False`` so it does not need an API key.
"""

from __future__ import annotations

import pytest

from agora_planner.mem0_adapter import Mem0Settings, OfficialMem0Store


def _mem0_available() -> bool:
    try:
        import mem0  # noqa: F401

        return True
    except Exception:
        return False


@pytest.mark.skipif(not _mem0_available(), reason="mem0ai is not installed")
def test_official_mem0_add_and_search(tmp_path):
    store = OfficialMem0Store(
        "pytest",
        settings=Mem0Settings(infer=False, embedding_dims=384),
        workdir=tmp_path,
    )
    try:
        store.add_texts(["The vault access code for facet alpha is K7Q2."])
        items = store.search("What is the vault access code for facet alpha?", top_k=3)
        assert items
        assert "K7Q2" in items[0].text
    finally:
        store.close()
