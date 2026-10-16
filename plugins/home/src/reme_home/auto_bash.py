"""Run allowlisted shell commands without an approval prompt."""

import asyncio
from functools import cache

from agentscope.message import TextBlock, ToolResultState
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionDecision
from agentscope.tool import Bash, ToolChunk

from reme.steps.base_step import BaseStep
from reme.steps.common.file_sandbox import file_sandbox_for_step

_READ_COMMANDS = {
    ":": {},
    "cat": {},
    "cd": {},
    "cmp": {},
    "diff": {"denied": ["--output"]},
    "echo": {},
    "false": {},
    "file": {
        "denied": ["-C", "--compile", "-p", "--preserve-date", "-z", "-Z", "--uncompress", "--uncompress-noreport"],
    },
    "find": {"denied": ["-delete", "-exec", "-execdir", "-fls", "-fprint", "-fprint0", "-fprintf", "-ok", "-okdir"]},
    "git": {
        "allow": ["status", "diff", "log", "show", "rev-parse", "ls-files"],
        "denied": [
            "--output",
            "--ext-diff",
            "--textconv",
            "--paginate",
            "--submodule",
            "--ignore-submodules",
            "--show-signature",
            "--remerge-diff",
            "--diff-merges",
            "--pretty",
            "--format",
        ],
    },
    "grep": {},
    "head": {},
    "ls": {},
    "pwd": {},
    "sort": {"denied": ["-o", "-T", "--output", "--compress-program", "--temporary-directory"]},
    "stat": {},
    "tail": {},
    "true": {},
    "wc": {},
    "which": {},
}
_SAFE_DESTINATIONS = {"/dev/null"}

# Trusted shell setup, applied only after validating the user's command.
# Git's configuration can launch helpers even for otherwise read-only commands.
_READ_ONLY_SETUP = r"""
unset GIT_EXTERNAL_DIFF GIT_TRACE GIT_TRACE_SETUP GIT_TRACE_PERFORMANCE
unset GIT_TRACE_PACK_ACCESS GIT_TRACE_PACKET GIT_TRACE_CURL GIT_TRACE2 GIT_TRACE2_EVENT GIT_TRACE2_PERF
export GIT_OPTIONAL_LOCKS=0 GIT_PAGER=/bin/cat GIT_NO_LAZY_FETCH=1 GIT_TERMINAL_PROMPT=0
sort() {
    # A large sort otherwise writes temporary files, even without -o.
    command sort -T /dev/null "$@"
}
git() {
    # These metadata queries do not read file contents through Git filters.
    if [ "$1" != rev-parse ] && [ "$1" != ls-files ] && \
        command git -c trace2.normalTarget=0 -c trace2.eventTarget=0 -c trace2.perfTarget=0 \
        config --get-regexp '^filter\..*\.(clean|smudge|process)$' > /dev/null; then
        echo 'Permission denied: Git repositories with external filters require unrestricted mode.' >&2
        return 1
    fi
    case "$1" in
        diff|log|show)
            reme_git_subcommand="$1"
            shift
            set -- "$reme_git_subcommand" --no-ext-diff --no-textconv --ignore-submodules=all "$@"
            ;;
        status) shift; set -- status --ignore-submodules=all "$@" ;;
    esac
    command git --no-pager -c core.fsmonitor=false -c diff.submodule=short \
        -c log.showSignature=false -c log.diffMerges=off -c format.pretty=medium \
        -c trace2.normalTarget=0 -c trace2.eventTarget=0 -c trace2.perfTarget=0 "$@"
}
"""


@cache
def _read_specs():
    """Build the fixed safecmd policy once, when read-only mode is used."""
    from safecmd import CmdSpec  # pylint: disable=import-outside-toplevel

    return {CmdSpec(name, denied=options.get("denied")) for name, options in _READ_COMMANDS.items()}


def _literal_shell(node: dict, command: str, quoted: bool = False) -> bool:
    """Reject expansions and syntax whose effects safecmd does not fully check."""
    from safecmd.bashxtract import word_text  # pylint: disable=import-outside-toplevel

    kind = node.get("Type")
    if kind in {"FuncDecl", "ForClause", "DeclClause", "ArithmExp", "ArithmCmd", "ProcSubst", "CmdSubst"}:
        return False
    if kind == "SglQuoted":
        return True
    quoted = quoted or kind == "DblQuoted"
    if kind == "Lit":
        value = node.get("Value", "")
        if "\\" in value or (not quoted and any(char in value for char in "*?[]{}~")):
            return False
    for redirect in node.get("Redirs", []):
        op = redirect.get("Op")
        destination = word_text(redirect.get("Word", {}), command)
        if op in {"<", "<<", "<<-"}:
            continue
        if op in {">&", "<&"} and (destination == "-" or destination.rstrip("-").isdigit()):
            continue
        if op not in {">", ">>", ">|", "<>", "&>", "&>>", ">&"} or destination != "/dev/null":
            return False
    for value in node.values():
        children = value if isinstance(value, list) else [value]
        if any(isinstance(child, dict) and not _literal_shell(child, command, quoted) for child in children):
            return False
    return True


def _allowed_arguments(tokens: list[str]) -> bool:
    """Supplement CmdSpec with subcommands, abbreviated flags and Git formats."""
    options = _READ_COMMANDS[tokens[0]]
    if "allow" in options and (len(tokens) < 2 or tokens[1] not in options["allow"]):
        return False
    # Signature placeholders invoke Git's configured GPG program.
    if tokens[0] == "git" and any("%G" in token for token in tokens[1:]):
        return False
    # getopt and Git also accept unambiguous long-option abbreviations.
    for token in tokens[1:]:
        flag = token.split("=", 1)[0]
        if flag.startswith("--") and len(flag) > 2:
            if any(denied.startswith(flag) for denied in options.get("denied", []) if denied.startswith("--")):
                return False
    return True


class AutoBash(Bash):
    """AgentScope Bash with an optional conservative read-only command guard.

    Shell expansions, function definitions and for loops are rejected because
    their runtime arguments or environment can differ from the parsed tokens.
    Git external filters require unrestricted mode; other Git helpers are
    disabled. Use path injection for OS file isolation.
    """

    name = "AutoBash"
    description = "Execute shell commands; read-only mode denies commands that may write or run other commands."
    input_schema = {
        **Bash.input_schema,
        "properties": {
            **Bash.input_schema["properties"],
            "timeout": {
                **Bash.input_schema["properties"]["timeout"],
                "description": "Optional timeout in milliseconds (default: 60000, max: 600000)",
                "default": 60000,
            },
        },
    }

    def __init__(self, *, is_read_only: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.is_read_only = is_read_only

    @staticmethod
    def _is_read_only(command: str) -> bool:
        """Use safecmd's Bash parser and a strict command policy."""
        if not isinstance(command, str) or not command.strip():
            return False
        # Variable and command expansion can hide a second operation in an
        # argument. Keep this first version literal while safecmd checks the
        # remaining shell syntax and nested commands.
        if "$" in command or "`" in command:
            return False
        try:
            from safecmd.bashxtract import (  # pylint: disable=import-outside-toplevel
                check_types,
                collect_ops,
                collect_redirects,
                parse_bash,
                visit_stmts,
            )
            from safecmd.core import normalize_dest, validate_cmd  # pylint: disable=import-outside-toplevel

            ast = parse_bash(command)
            check_types(ast)
            if not _literal_shell(ast, command):
                return False
            commands = visit_stmts(ast.get("Stmts", []), command)
            specs = _read_specs()
            if any(not validate_cmd(tokens, specs) or not _allowed_arguments(tokens) for tokens in commands):
                return False
            return "=" not in collect_ops(ast) and all(
                normalize_dest(destination) in _SAFE_DESTINATIONS for _, destination in collect_redirects(ast, command)
            )
        except Exception:  # pylint: disable=broad-exception-caught
            # Parser failures and missing shfmt must never allow execution.
            return False

    async def check_read_only(self, tool_input: dict) -> bool:
        """Report the same policy used by the execution guard."""
        return self.is_read_only and await asyncio.to_thread(self._is_read_only, tool_input.get("command", ""))

    async def check_permissions(self, tool_input: dict, _context: PermissionContext) -> PermissionDecision:
        """Resolve permission automatically, without an ASK decision."""
        if not self.is_read_only:
            return PermissionDecision(behavior=PermissionBehavior.ALLOW, message="Command allowed")
        allowed = await self.check_read_only(tool_input)
        return PermissionDecision(
            behavior=PermissionBehavior.ALLOW if allowed else PermissionBehavior.DENY,
            message=(
                "Read-only command allowed" if allowed else "Permission denied: only read-only commands are allowed"
            ),
        )

    async def call(self, command: str, description: str = "", timeout: int = 60000):
        """Apply the read-only guard when configured, then run the command."""
        if self.is_read_only and not await asyncio.to_thread(self._is_read_only, command):
            yield ToolChunk(
                content=[TextBlock(text="Permission denied: only read-only shell commands are allowed.")],
                state=ToolResultState.ERROR,
                is_last=True,
            )
            return
        if self.is_read_only:
            command = _READ_ONLY_SETUP + command
        async for chunk in super().call(command=command, description=description, timeout=timeout):
            yield chunk


class AutoBashStep(BaseStep):
    """Expose AgentScope Bash with configurable read-only mode as a ReMe job."""

    async def execute(self):
        """Run the command with the selected permission policy and backend."""
        assert self.context is not None
        command = self.context.get("command", "")
        timeout = self.context.get("timeout", self.kwargs.get("timeout", 60000))
        is_read_only = self.context.get("is_read_only", self.kwargs.get("is_read_only", True))
        if not isinstance(command, str) or not command.strip():
            self.context.response.success = False
            self.context.response.answer = "command is required"
            return self.context.response
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
            self.context.response.success = False
            self.context.response.answer = "timeout must be a positive integer in milliseconds"
            return self.context.response
        if not isinstance(is_read_only, bool):
            self.context.response.success = False
            self.context.response.answer = "is_read_only must be a boolean"
            return self.context.response

        try:
            backend = file_sandbox_for_step(self)
        except (TypeError, ValueError, RuntimeError) as exc:
            self.context.response.success = False
            self.context.response.answer = str(exc)
            return self.context.response

        tool = AutoBash(cwd=self.workspace_path, is_read_only=is_read_only, backend=backend)
        success = True
        texts = []
        async for chunk in tool.call(command=command, timeout=timeout):
            success = success and chunk.state != ToolResultState.ERROR
            texts.extend(block.text for block in chunk.content if isinstance(block, TextBlock))
        self.context.response.success = success
        self.context.response.answer = "".join(texts)
        return self.context.response
