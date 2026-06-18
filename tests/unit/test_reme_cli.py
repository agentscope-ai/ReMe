"""Tests for the ReMe CLI entry helpers."""

from reme import reme as reme_module


def test_call_server_passes_client_kwargs_to_client(monkeypatch, capsys):
    """CLI helper forwards connection options to the selected client."""
    seen = {}

    class FakeClient:
        """Async client stub that records call arguments."""

        def __init__(self, **kwargs):
            seen["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

        async def __call__(self, action: str, **kwargs):
            seen["action"] = action
            seen["payload"] = kwargs
            yield "ok"

    monkeypatch.setattr(reme_module.R, "get", lambda component_type, backend: FakeClient)

    async def run():
        await reme_module.call_server(
            "search",
            backend="http",
            host="127.0.0.2",
            port=2444,
            timeout=1.5,
            query="hello",
        )

    import asyncio

    asyncio.run(run())

    assert seen["client_kwargs"] == {"host": "127.0.0.2", "port": 2444, "timeout": 1.5}
    assert seen["action"] == "search"
    assert seen["payload"] == {"query": "hello"}
    assert capsys.readouterr().out == "ok\n"
