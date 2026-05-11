"""End-to-end tests for the reme2 MCP profiles.

Two suites — one per shipped profile (`expert`, `service`) — boot
`reme2.application.Application` against a temp vault and exercise every
MCP-exposed job through `app.run_job(...)`. We force `service.backend =
http` so the suites never block on stdio MCP transport; jobs are
called in-process.

Layout:

    _helpers.py          shared vault seed, app factory, response decoder
    test_expert.py       expert profile (every memory_* primitive surfaced)
    test_service.py      service profile (retrieve / remember / maintain)
    __main__.py          CLI: `python -m reme2.mcp.test`

The test functions are plain `async def check_<name>(app, ctx)` returning
True / raising. The CLI runner reuses one app per profile to keep the
boot cost out of the per-test budget.
"""
