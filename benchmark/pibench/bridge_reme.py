#!/usr/bin/env python3
"""
Bridge script: Connects ReMe agent to Pi-Bench Test Server.

Uses ReMe's AgentScope-based agent wrapper directly as a library,
with MCP integration to AppWorld and cross-session memory support.

Flow:
1. Poll Test Server /poll for user messages
2. Forward to ReMe agent (via AgentScope)
3. Extract reply text
4. Send reply back to Test Server POST /send
5. On session end (reset), save conversation as ReMe daily memory (non-blocking)
6. On every incoming user message, trigger a ReMe memory search and inject
   the relevant memories retrieved from previous sessions

Key design decisions:
- Memory saves are non-blocking (fire-and-forget asyncio tasks) so reset
  acknowledgments are sent immediately and don't time out.
- A pending-save tracker ensures the first message of a new session waits
  for any in-flight memory writes to complete before searching.
- User profile is loaded from data/{user_id}/profile.yaml and injected
  into every turn's system prompt.
- AgentScope session state is maintained via `resume` within a task,
  and cleared on reset for cross-task isolation.
- Memory search tuning: each search is capped at `--search-limit`
  results (default 3), weak BM25 hits below `--search-min-score`
  (default 2.0) are filtered, and a per-task `tool_context_id`
  enables ReMe's seen-chunk dedup so the same memory chunk is not
  re-injected on every turn of the same task.
- Persona isolation: the workspace defaults to a per-user subdirectory
  and an exclusive lock file guarantees that no two bridges can share
  one memory store at runtime.

Usage:
    python bridge_reme.py [--test-server-url URL] [--reme-dir DIR]
"""

import argparse
import asyncio
import fcntl
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("bridge_reme")


# ─── User Profile Loading ─────────────────────────────────────────────


def load_user_profile(data_root: str, user_id: str) -> str:
    """Load user profile YAML and return as formatted text for the system prompt.

    Handles the full Pi-Bench profile schema: role (with sub-sections),
    preferences, and long_term_goals.
    """
    profile_path = Path(data_root) / user_id / "profile.yaml"
    if not profile_path.exists():
        logger.warning("User profile not found: %s", profile_path)
        return ""

    with open(profile_path, "r", encoding="utf-8") as f:
        profile = yaml.safe_load(f)

    if not profile:
        return ""

    parts = []

    # Role section: contains the full persona description
    if "role" in profile and profile["role"]:
        role_text = str(profile["role"]).strip()
        if role_text:
            parts.append(f"## User Profile\n{role_text}")

    # Preferences section
    if "preferences" in profile and profile["preferences"]:
        prefs = profile["preferences"]
        if isinstance(prefs, dict):
            pref_lines = []
            for k, v in prefs.items():
                if v is not None and str(v).strip():
                    pref_lines.append(f"- {k}: {v}")
            if pref_lines:
                parts.append("## Preferences\n" + "\n".join(pref_lines))
        elif isinstance(prefs, str):
            parts.append(f"## Preferences\n{prefs}")

    # Long-term goals
    if "long_term_goals" in profile and profile["long_term_goals"]:
        goals = profile["long_term_goals"]
        if isinstance(goals, list):
            goal_lines = [f"- {g}" for g in goals if g]
            if goal_lines:
                parts.append("## Long-term Goals\n" + "\n".join(goal_lines))
        elif isinstance(goals, str):
            parts.append(f"## Long-term Goals\n{goals}")

    result = "\n\n".join(parts)
    logger.info(
        "Loaded profile for %s: %d chars, sections: %s",
        user_id,
        len(result),
        [k for k in ["role", "preferences", "long_term_goals"] if k in profile],
    )
    return result


def build_system_prompt(user_profile: str) -> str:
    """Build the system prompt for the ReMe agent with profile context."""
    base_prompt = """\
You are a proactive personal assistant agent in a long-horizon evaluation. Be thorough, anticipatory,
detail-oriented; use the user's profile, memory and tools proactively
(AppWorld via MCP; memory `search`/`daily_write`; file tools).

## HIDDEN-NEEDS PROTOCOL (MANDATORY)
Every task carries implicit needs the user does not state. Before each substantive response:
1. Derive the implicit needs of THIS task (method below), plus what the user's profile and past sessions imply.
2. Cover EVERY need explicitly and specifically in this response.
3. Anything you cannot cover now, you MUST still raise explicitly: one precise question or a concrete
next step targeting exactly that need. Generic closers do not count.

## HOW TO DERIVE IMPLICIT NEEDS
- Entities: for every item the task involves (a paper, product, person, account, case, event), cover the
attributes this user would need: what it is + key details, availability or cost, suitability/evaluation,
how to proceed, risks, and alternatives.
- Action completeness: if the task implies an action chain (prepare → execute → verify), cover every
stage, including verification and closing the loop.
- Context: apply everything the user's profile, constraints and past sessions imply (budget, size, format,
style, tools, deadlines) without being reminded.
- Structure: provide the format or verdict the user would expect (table, overall rating, pass/fail,
conclusion-first) whenever applicable.

## DELIVERABLE STRUCTURE
What (conclusion first) → Why → How → Risks (limits, fallbacks) → Next steps.

## STRICTNESS
An implicit need counts only with specific, detailed content or a concrete action — vague or generic
scores nothing. Deliver specifics in your FIRST response.
"""

    if user_profile:
        base_prompt += f"\n\n---\n\n{user_profile}\n"

    base_prompt += (
        "\n\n---\n\nAlways respond in the same language as the user's message. Use tools proactively to help the user."
    )
    return base_prompt


# ─── ReMe Bridge ──────────────────────────────────────────────────────


class ReMeBridge:
    """Bridge between Pi-Bench Test Server and ReMe agent."""

    def __init__(
        self,
        test_server_url: str = "http://localhost:9999",
        appworld_mcp_url: str = "http://localhost:10000/mcp",
        reme_dir: str = "",
        data_root: str = "data",
        user_id: str = "researcher",
        poll_timeout: int = 30,
        workspace_dir: str = "",
        model_name: str = "qwen3.6-plus",
        model_base_url: str = "",
        model_api_key: str = "",
        reme_port: int = 18765,
        search_limit: int = 3,
        search_min_score: float = 2.0,
    ):
        self.test_server_url = test_server_url.rstrip("/")
        self.appworld_mcp_url = appworld_mcp_url
        self.reme_dir = Path(reme_dir).resolve() if reme_dir else None
        self.data_root = Path(data_root)
        self.user_id = user_id
        self.poll_timeout = poll_timeout
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
        else:
            # Per-persona default so two bridges can never share a memory store.
            root = os.environ.get("REME_WORKSPACE_ROOT", "/tmp/reme_pibench_workspaces")
            self.workspace_dir = Path(root) / user_id
        self.model_name = model_name
        self.model_base_url = model_base_url
        self.model_api_key = model_api_key
        self.reme_port = reme_port
        self.search_limit = search_limit
        self.search_min_score = search_min_score
        # Task generation counter: rotated on every reset so the search dedup
        # context (tool_context_id) is scoped to a single task.
        self.task_seq = 0
        self._workspace_lock_fd: Optional[int] = None

        self.client: Optional[httpx.AsyncClient] = None
        self.running = False

        # ReMe components
        self.app = None
        self.agent_wrapper = None
        self.auto_memory_job = None
        self.search_job = None

        # Session state
        self.user_profile_text = ""
        self.session_messages: Dict[str, List[Dict]] = {}
        self.agent_session_id: Optional[str] = None

        # Non-blocking memory save tracking
        self._pending_memory_tasks: List[asyncio.Task] = []

    async def start(self):
        """Initialize the ReMe application and bridge components."""
        self.client = httpx.AsyncClient(timeout=300.0, trust_env=False)
        self.running = True

        # Enforce per-persona workspace isolation before anything else: an
        # exclusive lock guarantees no other bridge can use this memory store.
        self._acquire_workspace_lock()
        if self.workspace_dir.name != self.user_id:
            logger.warning(
                "Workspace basename %r != user_id %r; cross-persona isolation "
                "relies on each bridge having its own workspace_dir",
                self.workspace_dir.name,
                self.user_id,
            )
        logger.info(
            "Memory isolation: user=%s workspace=%s reme_port=%d",
            self.user_id,
            self.workspace_dir,
            self.reme_port,
        )

        # Load user profile
        self.user_profile_text = load_user_profile(str(self.data_root), self.user_id)
        logger.info("User profile loaded: %d chars", len(self.user_profile_text))

        # Initialize ReMe application
        await self._init_reme_app()

        logger.info(
            "Bridge started: test_server=%s appworld_mcp=%s user=%s model=%s",
            self.test_server_url,
            self.appworld_mcp_url,
            self.user_id,
            self.model_name,
        )

    def _acquire_workspace_lock(self):
        """Take an exclusive lock on the workspace (persona isolation guard)."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.workspace_dir / ".bridge.lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise SystemExit(
                f"Workspace {self.workspace_dir} is already locked by another "
                f"bridge process; each persona needs its own workspace_dir.",
            ) from exc
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} user={self.user_id}\n".encode())
        self._workspace_lock_fd = fd

    def _release_workspace_lock(self):
        if self._workspace_lock_fd is not None:
            try:
                fcntl.flock(self._workspace_lock_fd, fcntl.LOCK_UN)
                os.close(self._workspace_lock_fd)
            except OSError:
                pass
            self._workspace_lock_fd = None

    async def _init_reme_app(self):
        """Initialize the ReMe application with proper configuration."""
        # Add reme to Python path so imports work
        if self.reme_dir:
            reme_str = str(self.reme_dir)
            if reme_str not in sys.path:
                sys.path.insert(0, reme_str)

        try:
            from reme.config import resolve_app_config
            from reme.application import Application
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import 'reme'. Run the bridge with the ReMe venv "
                "python, or pass --reme-dir pointing to the ReMe repo root.",
            ) from exc

        # Set environment variables for ReMe LLM config expansion
        os.environ["LLM_MODEL_NAME"] = self.model_name
        if self.model_base_url:
            os.environ["LLM_BASE_URL"] = self.model_base_url
        if self.model_api_key:
            os.environ["LLM_API_KEY"] = self.model_api_key
        # Ensure BRAVE_SEARCH_API_KEY is set (required by some tools)
        if not os.environ.get("BRAVE_SEARCH_API_KEY"):
            os.environ["BRAVE_SEARCH_API_KEY"] = "dummy"

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # Load .env from reme dir if available
        environment = {}
        if self.reme_dir:
            env_path = self.reme_dir / ".env"
            if env_path.exists():
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, _, value = line.partition("=")
                            environment[key.strip()] = value.strip()

        # Override with explicit values
        if self.model_base_url:
            environment["LLM_BASE_URL"] = self.model_base_url
        if self.model_api_key:
            environment["LLM_API_KEY"] = self.model_api_key
        environment["LLM_MODEL_NAME"] = self.model_name

        # Use resolve_app_config to load default.yaml and merge overrides
        reme_config = resolve_app_config(
            log_config=False,
            workspace_dir=str(self.workspace_dir),
            service={"backend": "http", "host": "127.0.0.1", "port": self.reme_port},
            environment=environment,
        )

        try:
            self.app = Application(**reme_config)
            await self.app.start()

            # Get components
            self.agent_wrapper = self.app.context.components.get("agent_wrapper", {}).get("default")
            if self.agent_wrapper is None:
                raise RuntimeError("agent_wrapper component 'default' not found")

            # Get jobs for memory operations
            self.auto_memory_job = self.app.context.jobs.get("auto_memory")
            self.search_job = self.app.context.jobs.get("search")

            logger.info("ReMe initialized OK")
            logger.info("  agent_wrapper: %s", getattr(self.agent_wrapper, "name", "default"))
            logger.info("  auto_memory: %s", "yes" if self.auto_memory_job else "no")
            logger.info("  search: %s", "yes" if self.search_job else "no")
            logger.info("  total jobs: %d", len(self.app.context.jobs))

        except Exception as e:
            logger.exception("Failed to initialize ReMe: %s", e)
            raise

    async def stop(self):
        """Stop the bridge and cleanup."""
        self.running = False
        self._release_workspace_lock()

        # Wait for pending memory saves
        if self._pending_memory_tasks:
            logger.info("Waiting for %d pending memory saves...", len(self._pending_memory_tasks))
            for task in self._pending_memory_tasks:
                try:
                    await asyncio.wait_for(task, timeout=60.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning("Pending memory save timed out or failed: %s", e)

        if self.app:
            try:
                await self.app.close()
            except Exception as e:
                logger.warning("Error closing ReMe app: %s", e)
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Bridge stopped")

    def _create_mcp_client(self):
        """Create an MCP client for AppWorld."""
        from agentscope.mcp import MCPClient, HttpMCPConfig

        return MCPClient(
            name="AppWorld",
            is_stateful=False,
            mcp_config=HttpMCPConfig(
                url=self.appworld_mcp_url,
                timeout=120.0,
            ),
        )

    # ─── Memory Operations ──────────────────────────────────────────

    async def _wait_for_pending_memory_saves(self):
        """Wait for all in-flight memory save tasks to complete."""
        if not self._pending_memory_tasks:
            return
        logger.info(
            "Waiting for %d pending memory saves before search...",
            len(self._pending_memory_tasks),
        )
        tasks = self._pending_memory_tasks[:]
        self._pending_memory_tasks.clear()
        for task in tasks:
            try:
                await asyncio.wait_for(task, timeout=120.0)
            except asyncio.TimeoutError:
                logger.warning("Memory save task timed out (120s)")
            except Exception as e:
                logger.warning("Memory save task failed: %s", e)

    async def _search_memory(self, query: str, tool_context_id: str = "") -> str:
        """Search ReMe memory for relevant context from previous sessions."""
        if not self.search_job:
            return ""
        try:
            response = await self.search_job(
                query=query,
                limit=self.search_limit,
                min_score=self.search_min_score,
                tool_context_id=tool_context_id,
            )
            if response.success and response.answer:
                returned = response.metadata.get("counts", {}).get("returned", "?")
                logger.info(
                    "Memory search: hits=%s limit=%d min_score=%s ctx=%s",
                    returned,
                    self.search_limit,
                    self.search_min_score,
                    tool_context_id or "-",
                )
                return response.answer
        except Exception as e:
            logger.warning("Memory search failed: %s", e)
        return ""

    async def _do_save_session_memory(self, chat_id: str, messages: List[Dict]):
        """Actually perform the session memory save (runs as background task)."""
        if not self.auto_memory_job or not messages:
            return

        session_id = f"pibench_{chat_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Convert messages to auto_memory format
        memory_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            memory_messages.append(
                {
                    "role": role,
                    "name": msg.get("name", "user" if role == "user" else "assistant"),
                    "content": msg.get("content", ""),
                    "created_at": msg.get("timestamp", datetime.now().isoformat()),
                },
            )

        try:
            logger.info(
                "Saving session memory: chat_id=%s messages=%d",
                chat_id,
                len(memory_messages),
            )
            response = await self.auto_memory_job(
                messages=memory_messages,
                session_id=session_id,
                memory_hint=(
                    f"Pi-Bench evaluation session for task {chat_id}. "
                    f"User persona: {self.user_id}. "
                    f"Save key decisions, actions taken, important outcomes, "
                    f"and any user preferences or context that may be useful "
                    f"for future sessions."
                ),
            )
            if response.success:
                preview = (response.answer or "OK")[:200]
                logger.info("Session memory saved: %s", preview)
            else:
                logger.warning("Memory save returned unsuccessful: %s", response.answer)
        except Exception as e:
            logger.exception("Error saving session memory: %s", e)

    def _schedule_memory_save(self, chat_id: str, messages: List[Dict]):
        """Schedule a non-blocking memory save task."""
        if not self.auto_memory_job or not messages:
            return

        task = asyncio.create_task(
            self._do_save_session_memory(chat_id, messages),
            name=f"memory_save_{chat_id}",
        )
        self._pending_memory_tasks.append(task)

        # Clean up completed tasks from the tracking list
        self._pending_memory_tasks = [t for t in self._pending_memory_tasks if not t.done()]

    # ─── Message Processing ─────────────────────────────────────────

    async def process_message(self, _sender_id: str, chat_id: str, content: str) -> Optional[str]:
        """Process a user message through the ReMe agent."""
        # Track session messages for later memory save
        if chat_id not in self.session_messages:
            self.session_messages[chat_id] = []

        self.session_messages[chat_id].append(
            {
                "role": "user",
                "name": "user",
                "content": content,
                "timestamp": datetime.now().isoformat(),
            },
        )

        # Build system prompt with user profile
        system_prompt = build_system_prompt(self.user_profile_text)

        # Create MCP client for AppWorld
        mcp_client = self._create_mcp_client()

        # Determine which reme jobs to expose as tools
        job_tools = []
        if self.search_job:
            job_tools.append("search")
        if self.auto_memory_job:
            job_tools.extend(["auto_memory", "daily_write"])

        # On EVERY incoming user message, automatically trigger a ReMe
        # memory search and inject the relevant memories retrieved from
        # previous sessions. Memory is only surfaced through search
        # (relevance-filtered), never dumped wholesale. The in-progress
        # session is not in the store yet (saves happen on reset), so a
        # task can never retrieve its own partial content.
        memory_context = ""
        # Wait for any in-flight memory saves so the store is complete
        # before searching (no-op when nothing is pending).
        await self._wait_for_pending_memory_saves()
        if self.search_job:
            try:
                memory_context = await self._search_memory(
                    content,
                    tool_context_id=f"pibench_{self.user_id}_task_{self.task_seq}",
                )
                if memory_context:
                    logger.info("Found relevant memory: %d chars", len(memory_context))
            except Exception as e:
                logger.warning("Memory search failed: %s", e)

        try:
            # Prepend memory context if available
            user_message = content
            if memory_context:
                user_message = (
                    f"[Relevant memories from previous sessions]\n"
                    f"{memory_context}\n\n"
                    f"[Current user message]\n{content}"
                )

            # Call ReMe agent with MCP tools and memory tools
            reply_kwargs = {
                "system_prompt": system_prompt,
                "mcps": [mcp_client],
                "permission_mode": "bypass",
            }
            if job_tools:
                reply_kwargs["job_tools"] = job_tools

            # Resume existing session for multi-turn continuity within same task
            if self.agent_session_id:
                reply_kwargs["resume"] = self.agent_session_id

            result = await self.agent_wrapper.reply(user_message, **reply_kwargs)

            reply_text = result.get("result", "")
            session_id = result.get("session_id", "")

            if session_id:
                self.agent_session_id = session_id

            # Track the assistant reply
            self.session_messages[chat_id].append(
                {
                    "role": "assistant",
                    "name": "assistant",
                    "content": reply_text,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info("Reply: %d chars, session=%s", len(reply_text), session_id)
            return reply_text

        except Exception as e:
            logger.exception("Error processing message: %s", e)
            return None

    async def handle_reset(self, chat_id: str):
        """Handle session reset: schedule non-blocking memory save and clear state."""
        messages = self.session_messages.pop(chat_id, [])
        if messages:
            self._schedule_memory_save(chat_id, messages)
        # Clear agent session for cross-task isolation
        self.agent_session_id = None
        # New task boundary: rotate the search dedup context so memories can be
        # recalled again in the next task while repeats within a task are filtered.
        self.task_seq += 1

    # ─── Test Server Communication ──────────────────────────────────

    async def poll_test_server(self) -> Optional[List[Dict[str, Any]]]:
        """Poll Test Server for pending messages.

        Returns None on connection/response errors so the caller can back off;
        an empty list means a successful poll with no pending messages.
        """
        try:
            resp = await self.client.get(
                f"{self.test_server_url}/poll",
                params={"timeout": self.poll_timeout},
            )
            if resp.is_success:
                data = resp.json()
                messages = data.get("messages", [])
                if messages:
                    logger.info("Received %d messages", len(messages))
                return messages
            logger.warning("Poll returned HTTP %s", resp.status_code)
        except Exception as e:
            logger.warning("Poll error: %s", e)
        return None

    async def send_to_test_server(self, chat_id: str, content: str) -> bool:
        """Send reply back to Test Server."""
        payload = {
            "chat_id": chat_id,
            "content": content,
            "media": [],
            "meta": {},
        }
        try:
            resp = await self.client.post(
                f"{self.test_server_url}/send",
                json=payload,
            )
            if resp.is_success:
                logger.info("Sent reply: chat_id=%s len=%d", chat_id, len(content))
                return True
            logger.error("Failed to send: %s", resp.status_code)
        except Exception:
            logger.exception("Error sending to Test Server")
        return False

    # ─── Main Loop ──────────────────────────────────────────────────

    def _install_signal_handlers(self):
        """Install SIGTERM/SIGINT handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig_name in ("SIGTERM", "SIGINT"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                loop.add_signal_handler(sig, self._handle_shutdown_signal, sig_name)

    def _handle_shutdown_signal(self, sig_name: str):
        """Handle shutdown signal: stop the bridge loop gracefully."""
        logger.info("Received %s, initiating graceful shutdown...", sig_name)
        self.running = False

    async def run(self):
        """Main bridge loop."""
        await self.start()
        self._install_signal_handlers()

        consecutive_poll_errors = 0
        try:
            while self.running:
                messages = await self.poll_test_server()

                if messages is None:
                    # Back off on poll failure to avoid a tight error loop.
                    consecutive_poll_errors += 1
                    if consecutive_poll_errors in (1, 5, 20) or consecutive_poll_errors % 50 == 0:
                        logger.warning(
                            "Poll failure #%d, backing off",
                            consecutive_poll_errors,
                        )
                    await asyncio.sleep(min(2 ** min(consecutive_poll_errors, 5), 30))
                    continue
                consecutive_poll_errors = 0

                for msg in messages:
                    sender_id = msg.get("sender_id", "unknown")
                    chat_id = msg.get("chat_id", "default")
                    content = msg.get("content", "")

                    if not content:
                        continue

                    logger.info(
                        "Processing: sender=%s chat=%s len=%d",
                        sender_id,
                        chat_id,
                        len(content),
                    )

                    # Check for reset/new-session signal
                    if content.strip().lower() in ("reset", "new session", "/new"):
                        logger.info("Reset signal: chat_id=%s", chat_id)
                        await self.handle_reset(chat_id)
                        await self.send_to_test_server(chat_id, "New session started")
                        continue

                    # Forward to ReMe agent
                    reply = await self.process_message(sender_id, chat_id, content)

                    if reply:
                        await self.send_to_test_server(chat_id, reply)
                    else:
                        logger.warning("No reply for chat_id=%s", chat_id)
                        await self.send_to_test_server(
                            chat_id,
                            "[Error: Agent failed to generate response]",
                        )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Save any remaining session memories
            for cid in list(self.session_messages.keys()):
                messages = self.session_messages.pop(cid, [])
                if messages:
                    self._schedule_memory_save(cid, messages)
            # Wait for all pending memory saves to complete
            if self._pending_memory_tasks:
                logger.info(
                    "Waiting for %d pending memory saves to complete...",
                    len(self._pending_memory_tasks),
                )
                await self._wait_for_pending_memory_saves()
            await self.stop()


async def main():
    """CLI entrypoint: parse arguments and run the ReMe bridge."""
    # Ignore SIGHUP to prevent bridge from being killed (same fix as qwenpaw)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    parser = argparse.ArgumentParser(
        description="Bridge between Pi-Bench Test Server and ReMe agent",
    )
    parser.add_argument("--test-server-url", default="http://localhost:9999")
    parser.add_argument("--appworld-mcp-url", default="http://localhost:10000/mcp")
    parser.add_argument(
        "--reme-dir",
        default="",
        help="ReMe repo root. Optional when 'reme' is already importable "
        "(e.g. running inside the ReMe repo with its own venv).",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--user-id", default="researcher")
    parser.add_argument("--poll-timeout", type=int, default=30)
    parser.add_argument("--workspace-dir", default="")
    parser.add_argument("--model-name", default="qwen3.6-plus")
    parser.add_argument("--model-base-url", default="")
    parser.add_argument("--model-api-key", default="")
    parser.add_argument(
        "--reme-port",
        type=int,
        default=18765,
        help="Port for ReMe's internal HTTP service (must be unique per concurrently running bridge).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=3,
        help="Max memory chunks injected per user message.",
    )
    parser.add_argument(
        "--search-min-score",
        type=float,
        default=2.0,
        help="Min BM25 score for injected memory chunks.",
    )

    args = parser.parse_args()

    bridge = ReMeBridge(
        test_server_url=args.test_server_url,
        appworld_mcp_url=args.appworld_mcp_url,
        reme_dir=args.reme_dir,
        data_root=args.data_root,
        user_id=args.user_id,
        poll_timeout=args.poll_timeout,
        workspace_dir=args.workspace_dir,
        model_name=args.model_name,
        model_base_url=args.model_base_url,
        model_api_key=args.model_api_key,
        reme_port=args.reme_port,
        search_limit=args.search_limit,
        search_min_score=args.search_min_score,
    )

    await bridge.run()


if __name__ == "__main__":
    asyncio.run(main())
