"""Credentialed remote collectors: cursors, throttling, retries, and published packages."""

import json
from datetime import datetime, timezone

import httpx
import pytest

from reme.components.application_context import ApplicationContext
from reme.components.file_catalog import LocalFileCatalog
from reme.config.config_parser import resolve_app_config
from reme.schema import ResourceMetadata
from reme.steps.proactive.data_connectors import (
    BaseRemoteConnectorStep,
    GithubActivityStep,
    GithubNotificationsStep,
    ImapMailStep,
    NothingToPublish,
    TransientFetchError,
)
from reme.steps.proactive.data_connectors import github as github_module
from reme.steps.proactive.data_connectors import imap_mail as imap_module


def _app(tmp_path, **kwargs):
    """Build an ApplicationContext rooted in a temporary workspace."""
    return ApplicationContext(workspace_dir=str(tmp_path), **kwargs)


class _ProbeConnector(BaseRemoteConnectorStep):
    """Minimal remote collector exercising the shared plumbing without any network."""

    source = "probe"
    description = "Probe payloads."
    env_defaults = {"secret": "PROBE_SECRET"}
    attempts = 0

    def _fetch(self) -> list[str]:
        """Fail exactly as the configured plan asks, once per retryable kind."""
        self.attempts += 1
        payloads = []
        for item in self.kwargs.get("plan") or []:
            if item == "fail":
                raise RuntimeError("permanent failure")
            if item == "credential":
                self.credential("secret")
            if item == "transient":
                if self.attempts == 1:
                    raise TransientFetchError("temporarily unavailable")
                continue
            payloads.append(str(item))
        return payloads

    def collect(self, directory):
        """Stage one payload per planned item, or publish nothing when the plan is empty."""
        payloads = self.retry(self._fetch, description="probe fetch")
        if not payloads:
            raise NothingToPublish("nothing new since the previous poll")
        self.stage_state(cursor={"seen": len(payloads), "attempts": self.attempts})
        return [self.write_payload(directory, f"payloads/{name}.md", f"payload {name}") for name in payloads]


def _probe(tmp_path, **kwargs):
    """Instantiate the probe collector with test-friendly retry settings."""
    kwargs.setdefault("backend", "probe_step")
    kwargs.setdefault("retry_backoff_seconds", 0.1)
    return _ProbeConnector(app_context=_app(tmp_path), **kwargs)


def _message(uid, subject, body="hello there", sender="alice@example.com", size=None):
    """Render one raw RFC-822 message with an overridable declared size."""
    raw = (
        f"From: {sender}\r\nTo: me@example.com\r\nSubject: {subject}\r\n"
        f"Date: Wed, 0{uid % 9 + 1} Nov 2026 09:00:00 +0000\r\nMessage-ID: <{uid}@example.com>\r\n"
        "MIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\n"
        f"\r\n{body}\r\n"
    ).encode()
    return {"uid": uid, "raw": raw, "size": len(raw) if size is None else size}


class _FakeImapServer:
    """Answer just enough IMAP to drive a poll and record every command it received."""

    def __init__(self):
        """Start with an empty INBOX and no recorded traffic."""
        self.mailboxes: dict[str, dict] = {"INBOX": {"uidvalidity": "7", "messages": []}}
        self.calls: list[tuple] = []
        self.connections: list["_FakeConnection"] = []
        self.password = "secret"
        self.token = "ya29.token"
        self.logins: list[tuple[str, str]] = []
        self.fail_uids: set[int] = set()
        self.timeout = None

    def factory(self, host="", port=993, timeout=None):
        """Stand in for IMAP4_SSL and record the connection attempt."""
        self.calls.append(("connect", host, port))
        self.timeout = timeout
        connection = _FakeConnection(self)
        self.connections.append(connection)
        return connection


class _FakeConnection:
    """One fake IMAP session bound to its server state."""

    def __init__(self, server: _FakeImapServer):
        """Remember the shared server state for this session."""
        self.server = server
        self.selected: str | None = None
        self.logged_out = 0

    def login(self, user, password):
        """Accept only the configured password and record the attempt."""
        self.server.calls.append(("login", user))
        self.server.logins.append((user, password))
        if password != self.server.password:
            raise RuntimeError("authentication failed")
        return "OK", [b"logged in"]

    def authenticate(self, mechanism, authobject):
        """Run the SASL callback and require the configured bearer token."""
        self.server.calls.append(("authenticate", mechanism))
        response = authobject(b"")
        if self.server.token.encode() not in response:
            raise RuntimeError("authentication failed")
        return "OK", [b"authenticated"]

    def select(self, mailbox, readonly=False):
        """Select a known mailbox and record whether it was read-only."""
        self.server.calls.append(("select", mailbox, readonly))
        if mailbox not in self.server.mailboxes:
            return "NO", [b"unknown mailbox"]
        self.selected = mailbox
        return "OK", [str(len(self.server.mailboxes[mailbox]["messages"])).encode()]

    def response(self, name):
        """Serve the UIDVALIDITY untagged response for the selected mailbox."""
        if name.upper() == "UIDVALIDITY" and self.selected:
            return "OK", [str(self.server.mailboxes[self.selected]["uidvalidity"]).encode()]
        return "OK", [None]

    def uid(self, command, *args):
        """Answer UID SEARCH and UID FETCH from the canned mailbox contents."""
        self.server.calls.append(("uid", command.upper(), *[str(arg) for arg in args]))
        mailbox = self.server.mailboxes[self.selected]
        if command.upper() == "SEARCH":
            found = " ".join(str(item["uid"]) for item in self._matching(mailbox, args))
            return "OK", [found.encode()]
        if command.upper() != "FETCH":
            raise AssertionError(f"unexpected IMAP command {command}")
        uids = [int(item) for item in str(args[0]).split(",")]
        section = str(args[1]) if len(args) > 1 else ""
        if "RFC822.SIZE" in section:
            known = {item["uid"]: item for item in mailbox["messages"]}
            return "OK", [f"1 (UID {uid} RFC822.SIZE {known[uid]['size']})".encode() for uid in uids if uid in known]
        payloads = []
        for uid in uids:
            if uid in self.server.fail_uids:
                return "NO", [b"fetch failed"]
            known = {item["uid"]: item for item in mailbox["messages"]}.get(uid)
            if known is None:
                continue
            raw = known["raw"].split(b"\r\n\r\n")[0] + b"\r\n\r\n" if "HEADER" in section else known["raw"]
            payloads.append(((f"1 (UID {uid} BODY[] {{{len(raw)}}}").encode(), raw))
        payloads.append(b")")
        return "OK", payloads

    @staticmethod
    def _matching(mailbox, args):
        """Apply the SEARCH criteria this suite issues."""
        criteria = [str(arg) for arg in args]
        items = list(mailbox["messages"])
        if "UNSEEN" in criteria:
            items = [item for item in items if item["uid"] in mailbox.get("unseen", [])]
        if "UID" in criteria:
            low = int(criteria[criteria.index("UID") + 1].split(":")[0])
            items = [item for item in items if item["uid"] >= low]
        return items

    def logout(self):
        """Record that the poll released its connection."""
        self.logged_out += 1
        self.server.calls.append(("logout",))
        return "BYE", [b"logging out"]


@pytest.fixture(name="server")
def _server(monkeypatch):
    state = _FakeImapServer()
    monkeypatch.setattr(imap_module, "IMAP4_SSL", state.factory)
    return state


def _mail(tmp_path, **kwargs):
    """Instantiate the IMAP collector pointed at the fake server's host."""
    kwargs.setdefault("backend", "imap_mail_step")
    kwargs.setdefault("host", "imap.example.com")
    kwargs.setdefault("user", "me@example.com")
    kwargs.setdefault("password", "secret")
    kwargs.setdefault("retry_backoff_seconds", 0.1)
    return ImapMailStep(app_context=_app(tmp_path), **kwargs)


class _FakeGithub:
    """Replay canned GitHub REST pages and record every request."""

    def __init__(self):
        """Start with no routes and no recorded requests."""
        self.routes: dict[str, list[httpx.Response]] = {}
        self.requests: list[httpx.Request] = []

    def route(self, path: str, *responses: httpx.Response) -> None:
        """Queue the responses one path should replay in order."""
        self.routes.setdefault(path, []).extend(responses)

    def transport(self) -> httpx.MockTransport:
        """Build the transport that replays queued pages and records requests."""

        def handler(request: httpx.Request) -> httpx.Response:
            """Replay the queued response for this path, or a 404."""
            self.requests.append(request)
            queued = self.routes.get(request.url.path)
            if not queued:
                return httpx.Response(404, json={"message": "Not Found"})
            return queued.pop(0) if len(queued) > 1 else queued[0]

        return httpx.MockTransport(handler)


@pytest.fixture(name="github")
def _github(monkeypatch):
    fake = _FakeGithub()
    real_client = httpx.Client

    def client_factory(**kwargs):
        """Inject the fake transport into every client the collector opens."""
        return real_client(transport=fake.transport(), **kwargs)

    monkeypatch.setattr(github_module.httpx, "Client", client_factory)
    return fake


def _notification(
    title="feat(pa): bind reminder tools",
    repo="agentscope-ai/ReMe",
    number=714,
    kind="PullRequest",
    updated="2026-11-05T10:00:00Z",
):
    """Build one GitHub notification payload."""
    return {
        "id": "1",
        "repository": {"full_name": repo},
        "reason": "review_requested",
        "unread": True,
        "updated_at": updated,
        "subject": {"title": title, "url": f"https://api.github.com/repos/{repo}/pulls/{number}", "type": kind},
        "url": f"https://api.github.com/repos/{repo}/notifications/threads/1",
    }


def _issue(number=42, login="wce", updated="2026-11-05T08:00:00Z", pull=False):
    """Build one GitHub issue or pull request payload."""
    return {
        "number": number,
        "title": f"item {number}",
        "state": "open",
        "user": {"login": login},
        "assignees": [],
        "labels": [{"name": "bug"}],
        "comments": 2,
        "updated_at": updated,
        "html_url": f"https://github.com/agentscope-ai/ReMe/{'pull' if pull else 'issues'}/{number}",
        **({"pull_request": {"url": "https://api.github.com/x"}} if pull else {}),
    }


def _github_step(cls, tmp_path, monkeypatch, **kwargs):
    """Instantiate a GitHub collector with a token in the environment."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    kwargs.setdefault("backend", cls.__name__)
    kwargs.setdefault("retry_backoff_seconds", 0.1)
    return cls(app_context=_app(tmp_path), **kwargs)


@pytest.mark.asyncio
async def test_empty_poll_records_the_poll_without_publishing(tmp_path):
    """A poll with nothing new stays a successful Job run and leaves no package behind."""
    step = _probe(tmp_path, plan=[])
    response = await step()
    assert response.success is True
    assert response.answer["path"] is None
    assert response.answer["skipped"] == "nothing new since the previous poll"
    assert not list((tmp_path / "resource").rglob("*"))
    assert not list(tmp_path.glob(".reme-resource-*"))
    state = json.loads(step.state_path().read_text(encoding="utf-8"))
    assert state["polls"] == 1
    assert state["last_poll_at"]
    assert "cursor" not in state


@pytest.mark.asyncio
async def test_cursor_is_persisted_only_together_with_a_published_package(tmp_path):
    """A failed collection must leave the previous cursor untouched so the window is re-read."""
    failing = _probe(tmp_path, plan=["fail"], max_retries=0)
    with pytest.raises(RuntimeError, match="permanent failure"):
        await failing()
    assert not failing.state_path().exists()

    step = _probe(tmp_path, plan=["alpha"])
    response = await step()
    manifest = tmp_path / response.answer["path"]
    assert manifest.is_file()
    state = json.loads(step.state_path().read_text(encoding="utf-8"))
    assert state["cursor"] == {"seen": 1, "attempts": 1}
    assert state["connector"] == "probe_step"
    assert ResourceMetadata.model_validate_json(manifest.read_text(encoding="utf-8")).source == "probe"


@pytest.mark.asyncio
async def test_transient_failures_are_retried_inside_one_poll(tmp_path):
    """A retryable provider error is retried with backoff instead of failing the round."""
    step = _probe(tmp_path, plan=["transient", "alpha"], max_retries=2)
    response = await step()
    assert response.answer["files"] == [{"path": "payloads/alpha.md"}]
    assert step.attempts == 2
    assert json.loads(step.state_path().read_text(encoding="utf-8"))["cursor"]["attempts"] == 2


@pytest.mark.asyncio
async def test_missing_credentials_skip_the_poll_without_failing_the_job(tmp_path, server, monkeypatch):
    """An unconfigured source warns and publishes nothing rather than breaking the cron Job."""
    monkeypatch.delenv("IMAP_HOST", raising=False)
    step = ImapMailStep(app_context=_app(tmp_path), backend="imap_mail_step", user="me@example.com")
    response = await step()
    assert response.success is True
    assert response.answer["path"] is None
    assert response.answer["skipped"].startswith("missing_credentials")
    assert "IMAP_HOST" in response.answer["skipped"]
    assert not list((tmp_path / "resource").rglob("*"))
    assert not list(tmp_path.glob(".reme-resource-*"))
    assert not step.state_path().exists()
    assert server.connections == []


@pytest.mark.asyncio
async def test_a_rejected_login_fails_the_round_loudly(tmp_path, server):
    """A wrong authorization code surfaces as an error, unlike a missing one."""
    server.password = "another-code"
    step = _mail(tmp_path, max_retries=0)
    with pytest.raises(RuntimeError, match="authentication failed"):
        await step()
    assert not step.state_path().exists()


@pytest.mark.asyncio
async def test_a_throttled_poll_leaves_the_cursor_alone(tmp_path):
    """Refusing an early poll must not extend the interval that refused it."""
    first = await _probe(tmp_path, plan=["alpha"], min_interval_minutes=30)()
    assert first.answer["path"]
    persisted = _probe(tmp_path).state_path().read_text(encoding="utf-8")

    step = _probe(tmp_path, plan=["alpha"], min_interval_minutes=30)
    response = await step()
    assert response.answer["path"] is None
    assert "min_interval_minutes" in response.answer["skipped"]
    assert step.state_path().read_text(encoding="utf-8") == persisted
    assert json.loads(persisted)["polls"] == 1


def test_payload_helpers_reject_empty_and_unsafe_names(tmp_path):
    """Payload paths stay inside the staging directory and never publish an empty file."""
    step = _probe(tmp_path)
    with pytest.raises(ValueError, match="empty payload"):
        step.write_payload(tmp_path, "blank.md", "   ")
    with pytest.raises(ValueError, match="unsafe payload path"):
        step.write_payload(tmp_path, "../escape.md", "text")
    entry = step.write_payload(tmp_path, "messages/Re: hello world.md", "text")
    assert entry == {"path": "messages/Re_hello_world.md"}
    assert (tmp_path / "messages/Re_hello_world.md").read_text(encoding="utf-8") == "text\n"
    assert step.truncate("x" * 20, 10).startswith("x" * 10)
    assert "truncated" in step.truncate("x" * 20, 10)


@pytest.mark.asyncio
async def test_mail_poll_publishes_one_package_and_advances_the_uid_cursor(tmp_path, server):
    """Every new message becomes one Markdown payload in a single dated resource package."""
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First"), _message(12, "Second")]
    step = _mail(tmp_path)
    response = await step()
    manifest = tmp_path / response.answer["path"]
    metadata = ResourceMetadata.model_validate_json(manifest.read_text(encoding="utf-8"))
    assert metadata.source == "mail"
    assert "2 messages" in metadata.description
    assert [item.path for item in metadata.files] == ["messages/001-INBOX-11.md", "messages/002-INBOX-12.md"]
    body = (manifest.parent / "messages/001-INBOX-11.md").read_text(encoding="utf-8")
    assert body.startswith("# First")
    assert "alice@example.com" in body and "hello there" in body
    assert json.loads(step.state_path().read_text(encoding="utf-8"))["cursor"]["mailboxes"]["INBOX"] == {
        "uidvalidity": "7",
        "last_uid": 12,
    }
    assert ("select", "INBOX", "True") in [(name, *map(str, rest)) for name, *rest in server.calls if name == "select"]
    assert server.connections[0].logged_out == 1
    assert server.timeout == 30.0


@pytest.mark.asyncio
async def test_a_secret_from_the_environment_reaches_the_server_verbatim(tmp_path, server, monkeypatch):
    """Credentials skip YAML on purpose: a number-looking value there would be coerced to a float."""
    monkeypatch.setenv("IMAP_PASSWORD", "1e5")
    server.password = "1e5"
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First")]
    response = await _mail(tmp_path, password=None)()
    assert response.answer["path"]
    assert server.logins == [("me@example.com", "1e5")]


@pytest.mark.asyncio
async def test_a_second_poll_reads_only_newer_uids(tmp_path, server):
    """The stored cursor narrows the next search instead of re-reading the mailbox."""
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First")]
    await _mail(tmp_path)()
    server.mailboxes["INBOX"]["messages"].append(_message(12, "Second"))
    server.calls.clear()

    step = _mail(tmp_path)
    response = await step()
    searches = [rest for name, *rest in server.calls if name == "uid" and rest[0] == "SEARCH"]
    assert searches == [["SEARCH", "UID", "12:*"]]
    assert [
        item.path
        for item in ResourceMetadata.model_validate_json(
            (tmp_path / response.answer["path"]).read_text(encoding="utf-8"),
        ).files
    ] == [
        "messages/001-INBOX-12.md",
    ]


@pytest.mark.asyncio
async def test_a_changed_uidvalidity_resets_the_mailbox_cursor(tmp_path, server):
    """A renumbered mailbox falls back to a bounded date window instead of trusting stale UIDs."""
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First")]
    await _mail(tmp_path)()
    server.mailboxes["INBOX"]["uidvalidity"] = "9"
    server.calls.clear()

    step = _mail(tmp_path, initial_window_days=3)
    await step()
    searches = [rest for name, *rest in server.calls if name == "uid" and rest[0] == "SEARCH"]
    assert searches and searches[-1][1] == "SINCE"
    state = json.loads(step.state_path().read_text(encoding="utf-8"))
    assert state["cursor"]["mailboxes"]["INBOX"]["uidvalidity"] == "9"


@pytest.mark.asyncio
async def test_an_unreadable_message_keeps_the_cursor_behind_it(tmp_path, server):
    """A message that cannot be fetched is retried later rather than skipped forever."""
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First"), _message(12, "Second")]
    server.fail_uids = {12}
    step = _mail(tmp_path)
    response = await step()
    manifest = tmp_path / response.answer["path"]
    assert [item.path for item in ResourceMetadata.model_validate_json(manifest.read_text(encoding="utf-8")).files] == [
        "messages/001-INBOX-11.md",
    ]
    assert json.loads(step.state_path().read_text(encoding="utf-8"))["cursor"]["mailboxes"]["INBOX"]["last_uid"] == 11


@pytest.mark.asyncio
async def test_the_message_budget_spreads_over_mailboxes(tmp_path, server):
    """Mailboxes are drained oldest first and the unvisited one keeps its previous cursor."""
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First"), _message(12, "Second")]
    server.mailboxes["Archive"] = {"uidvalidity": "3", "messages": [_message(21, "Old")]}
    first = _mail(tmp_path, mailboxes=["INBOX", "Archive"], max_messages=1)
    response = await first()
    assert response.answer["files"] == [{"path": "messages/001-INBOX-11.md"}]
    state = json.loads(first.state_path().read_text(encoding="utf-8"))["cursor"]["mailboxes"]
    assert state == {"INBOX": {"uidvalidity": "7", "last_uid": 11}}

    server.calls.clear()
    second = _mail(tmp_path, mailboxes=["INBOX", "Archive"], max_messages=2)
    await second()
    state = json.loads(second.state_path().read_text(encoding="utf-8"))["cursor"]["mailboxes"]
    assert state == {
        "INBOX": {"uidvalidity": "7", "last_uid": 12},
        "Archive": {"uidvalidity": "3", "last_uid": 21},
    }


@pytest.mark.asyncio
async def test_an_oversized_message_is_skipped(tmp_path, server):
    """Huge mail is left out so one message cannot blow up a package."""
    server.mailboxes["INBOX"]["messages"] = [_message(11, "Huge", size=5 * 1024 * 1024), _message(12, "Small")]
    step = _mail(tmp_path, max_message_bytes=1024)
    response = await step()
    manifest = tmp_path / response.answer["path"]
    assert response.answer["files"] == [{"path": "messages/001-INBOX-12.md"}]
    assert (manifest.parent / "messages/001-INBOX-12.md").read_text(encoding="utf-8").startswith("# Small")


@pytest.mark.asyncio
async def test_oauth_login_sends_a_bearer_token(tmp_path, server, monkeypatch):
    """A configured access token authenticates without a password reaching the connection."""
    monkeypatch.setenv("IMAP_ACCESS_TOKEN", "ya29.token")
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First")]
    step = _mail(tmp_path, auth_mode="oauthbearer", password=None)
    response = await step()
    assert response.answer["path"]
    assert ("authenticate", "oauthbearer") in server.calls
    assert not any(name == "login" for name, *_ in server.calls)


@pytest.mark.asyncio
async def test_notifications_publish_a_digest_and_store_the_server_cursor(tmp_path, github, monkeypatch):
    """One digest file is published and the Last-Modified header becomes the next cursor."""
    github.route(
        "/notifications",
        httpx.Response(
            200,
            json=[_notification()],
            headers={"Last-Modified": "Wed, 05 Nov 2026 10:00:00 GMT", "X-RateLimit-Remaining": "42"},
        ),
    )
    step = _github_step(GithubNotificationsStep, tmp_path, monkeypatch, backend="github_notifications_step")
    response = await step()
    manifest = tmp_path / response.answer["path"]
    metadata = ResourceMetadata.model_validate_json(manifest.read_text(encoding="utf-8"))
    assert metadata.source == "github"
    assert [item.path for item in metadata.files] == ["notifications.md"]
    digest = (manifest.parent / "notifications.md").read_text(encoding="utf-8")
    assert "## agentscope-ai/ReMe" in digest
    assert "https://github.com/agentscope-ai/ReMe/pull/714" in digest
    assert "review_requested" in digest
    assert json.loads(step.state_path().read_text(encoding="utf-8"))["cursor"]["since"] == "2026-11-05T10:00:00Z"
    assert "since=" in str(github.requests[0].url)
    assert github.requests[0].headers["Authorization"] == "Bearer ghp_test"


@pytest.mark.asyncio
async def test_filtered_notifications_advance_the_cursor_without_a_package(tmp_path, github, monkeypatch):
    """Ignoring a repository is a decision, so its window closes even though nothing is published."""
    github.route(
        "/notifications",
        httpx.Response(
            200,
            json=[_notification(repo="noise/spam")],
            headers={
                "Last-Modified": "Wed, 05 Nov 2026 11:00:00 GMT",
            },
        ),
    )
    step = _github_step(
        GithubNotificationsStep,
        tmp_path,
        monkeypatch,
        backend="github_notifications_step",
        exclude_repos="noise/spam",
    )
    response = await step()
    assert response.answer["path"] is None
    assert "no GitHub notifications" in response.answer["skipped"]
    assert json.loads(step.state_path().read_text(encoding="utf-8"))["cursor"]["since"] == "2026-11-05T11:00:00Z"


@pytest.mark.asyncio
async def test_a_repeated_boundary_notification_is_published_once(tmp_path, github, monkeypatch):
    """GitHub answers `since` inclusively, so the cursor's own item must not return as a new package."""
    github.route(
        "/notifications",
        httpx.Response(200, json=[_notification()], headers={"Last-Modified": "Wed, 05 Nov 2026 10:00:00 GMT"}),
        httpx.Response(200, json=[_notification()], headers={"Last-Modified": "Wed, 05 Nov 2026 10:00:00 GMT"}),
        httpx.Response(
            200,
            json=[_notification(updated="2026-11-05T12:00:00Z")],
            headers={"Last-Modified": "Wed, 05 Nov 2026 12:00:00 GMT"},
        ),
    )

    def build():
        """Rebuild the collector so every poll re-reads the cursor the previous poll persisted."""
        return _github_step(
            GithubNotificationsStep,
            tmp_path,
            monkeypatch,
            backend="github_notifications_step",
        )

    assert (await build()()).answer["path"]
    repeat = await build()()
    assert repeat.answer["path"] is None
    assert "no GitHub notifications" in repeat.answer["skipped"]
    cursor = json.loads(build().state_path().read_text(encoding="utf-8"))["cursor"]
    assert cursor["seen"] == ["2026-11-05T10:00:00Z|1"]

    updated = await build()()
    assert updated.answer["path"], "a newer update to the same thread is new information"
    assert json.loads(build().state_path().read_text(encoding="utf-8"))["cursor"]["seen"] == [
        "2026-11-05T10:00:00Z|1",
        "2026-11-05T12:00:00Z|1",
    ]


@pytest.mark.asyncio
async def test_repeated_activity_items_are_published_once(tmp_path, github, monkeypatch):
    """An inclusive per-repository cursor must not republish the same issue version."""
    github.route("/user", httpx.Response(200, json={"login": "wce"}))
    github.route("/repos/agentscope-ai/ReMe/issues", httpx.Response(200, json=[_issue(42)]))

    def build():
        """Rebuild the collector so every poll re-reads the cursor the previous poll persisted."""
        return _github_step(
            GithubActivityStep,
            tmp_path,
            monkeypatch,
            backend="github_activity_step",
            repos=["agentscope-ai/ReMe"],
        )

    assert (await build()()).answer["path"]
    repeat = await build()()
    assert repeat.answer["path"] is None
    cursor = json.loads(build().state_path().read_text(encoding="utf-8"))["cursor"]["repos"]
    assert cursor["agentscope-ai/ReMe"]["seen"] == ["2026-11-05T08:00:00Z|42"]
    assert cursor["agentscope-ai/ReMe"]["last_collected"] == 0


@pytest.mark.asyncio
async def test_a_throttled_github_response_is_retried(tmp_path, github, monkeypatch):
    """A rate-limit rejection waits for the advised delay and then succeeds in the same poll."""
    github.route(
        "/notifications",
        httpx.Response(403, json={"message": "API rate limit exceeded for user"}, headers={"Retry-After": "0"}),
        httpx.Response(200, json=[_notification()], headers={"Last-Modified": "Wed, 05 Nov 2026 10:00:00 GMT"}),
    )
    step = _github_step(GithubNotificationsStep, tmp_path, monkeypatch, backend="github_notifications_step")
    response = await step()
    assert response.answer["path"]
    assert len(github.requests) == 2


@pytest.mark.asyncio
async def test_activity_requires_configured_repositories(tmp_path, github, monkeypatch):
    """Without a repository list the collector publishes nothing and issues no request."""
    step = _github_step(GithubActivityStep, tmp_path, monkeypatch, backend="github_activity_step")
    response = await step()
    assert response.answer["path"] is None
    assert "no repositories configured" in response.answer["skipped"]
    assert github.requests == []


@pytest.mark.asyncio
async def test_activity_keeps_one_cursor_per_repository_and_flags_the_viewer(tmp_path, github, monkeypatch):
    """Per-repository cursors advance to the newest update and involvement is annotated."""
    github.route("/user", httpx.Response(200, json={"login": "wce"}))
    github.route(
        "/repos/agentscope-ai/ReMe/issues",
        httpx.Response(
            200,
            json=[_issue(42, login="alice"), _issue(43, login="wce", updated="2026-11-05T09:00:00Z", pull=True)],
        ),
    )
    step = _github_step(
        GithubActivityStep,
        tmp_path,
        monkeypatch,
        backend="github_activity_step",
        repos="agentscope-ai/ReMe",
    )
    response = await step()
    manifest = tmp_path / response.answer["path"]
    digest = (manifest.parent / "activity.md").read_text(encoding="utf-8")
    assert "[Issue #42]" in digest and "[PR #43]" in digest
    assert "involves you as author" in digest
    state = json.loads(step.state_path().read_text(encoding="utf-8"))
    assert state["cursor"]["repos"]["agentscope-ai/ReMe"]["since"] == "2026-11-05T09:00:00Z"
    assert state["cursor"]["viewer"] == "wce"


@pytest.mark.asyncio
async def test_activity_can_restrict_items_to_the_authenticated_login(tmp_path, github, monkeypatch):
    """The involvement filter drops items the user neither authored nor is assigned to."""
    github.route("/user", httpx.Response(200, json={"login": "wce"}))
    github.route("/repos/agentscope-ai/ReMe/issues", httpx.Response(200, json=[_issue(42, login="alice")]))
    step = _github_step(
        GithubActivityStep,
        tmp_path,
        monkeypatch,
        backend="github_activity_step",
        repos=["agentscope-ai/ReMe"],
        only_involving_you=True,
    )
    response = await step()
    assert response.answer["path"] is None
    assert "no issue or pull request updates" in response.answer["skipped"]


@pytest.mark.asyncio
async def test_collected_packages_reach_the_pa_resource_catalog(tmp_path, server):
    """A published mail package is exactly what the PA collector's resource_changes tool reports."""
    server.mailboxes["INBOX"]["messages"] = [_message(11, "First")]
    app = _app(tmp_path)
    catalog = LocalFileCatalog(
        name="resource",
        app_context=app,
        watch_dirs=["resource_dir"],
        watch_suffixes=[],
        watch_filenames=["metadata.json"],
        ttl_days=7,
    )
    await catalog.load()
    assert await catalog.changes(limit=10) == []

    response = await _mail(tmp_path)()
    path = response.answer["path"]
    assert await catalog.changes(limit=10) == [path]
    assert await catalog.processed([path]) is True
    assert await catalog.changes(limit=10) == []


def test_remote_config_extends_the_proactive_workspace():
    """The credentialed collectors join the existing PA loop without redefining it."""
    config = resolve_app_config(log_config=False, config="proactive_remote")
    assert [step["backend"] for step in config["jobs"]["mail_capture_cron"]["steps"]] == ["imap_mail_step"]
    assert [step["backend"] for step in config["jobs"]["github_capture_cron"]["steps"]] == [
        "github_notifications_step",
        "github_activity_step",
    ]
    assert config["jobs"]["pa_loop"]["steps"][0]["default_wakeup_minutes"] == 5
    assert config["jobs"]["resource_watch_loop"]["file_catalog"] == "resource"
    assert config["components"]["file_catalog"]["resource"]["watch_filenames"] == ["metadata.json"]
    assert datetime.now(timezone.utc).year > 2025
