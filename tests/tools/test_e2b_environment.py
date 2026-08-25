"""Focused tests for the E2B terminal backend."""

from __future__ import annotations

import json
import sys
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


class SandboxFailure(Exception):
    pass


class SandboxMissing(SandboxFailure):
    pass


class ApiRateLimited(SandboxFailure):
    pass


class AuthenticationFailed(Exception):
    pass


class RemoteFileMissing(Exception):
    pass


class CommandFailed(Exception):
    def __init__(self, *, stdout="", stderr="", exit_code=1, error=None):
        super().__init__(stderr)
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.error = error


@dataclass
class CommandResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    error: str | None = None


class RecordingCommandHandle:
    def __init__(self, result: CommandResult | None = None):
        self.result = result or CommandResult()
        self.kill_calls = 0

    def wait(self, on_stdout=None, on_stderr=None):
        if self.result.stdout and on_stdout:
            on_stdout(self.result.stdout)
        if self.result.stderr and on_stderr:
            on_stderr(self.result.stderr)
        if self.result.exit_code:
            raise CommandFailed(
                stdout=self.result.stdout,
                stderr=self.result.stderr,
                exit_code=self.result.exit_code,
                error=self.result.error,
            )
        return self.result

    def kill(self):
        self.kill_calls += 1
        return True


class RecordingCommands:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.run_hook = None

    def run(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        if self.run_hook is not None:
            return self.run_hook(command, kwargs)
        result = CommandResult()
        return RecordingCommandHandle(result) if kwargs.get("background") else result


class FakeStreamReader:
    """Chunked download stream mirroring the SDK's FileStreamReader shape."""

    def __init__(self, payload: bytes, chunk_size: int = 7):
        self._chunks = [
            payload[i : i + chunk_size] for i in range(0, len(payload), chunk_size)
        ]
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self):
        self.closed = True


class RecordingFiles:
    def __init__(self):
        self.write_calls: list[tuple[str, bytes]] = []
        self.write_files_calls: list[list[dict]] = []
        self.remove_calls: list[str] = []
        self.remove_effects: dict[str, Exception] = {}
        self.read_payload = b"archive"
        self.stream_readers: list[FakeStreamReader] = []

    def write(self, path: str, data: bytes):
        self.write_calls.append((path, data))

    def write_files(self, files: list[dict]):
        self.write_files_calls.append(files)

    def remove(self, path: str):
        self.remove_calls.append(path)
        effect = self.remove_effects.get(path)
        if effect:
            raise effect

    def read(self, path: str, *, format: str):
        assert format == "stream"
        reader = FakeStreamReader(self.read_payload)
        self.stream_readers.append(reader)
        return reader


class SandboxSession:
    def __init__(self, sandbox_id: str):
        self.sandbox_id = sandbox_id
        self.commands = RecordingCommands()
        self.files = RecordingFiles()
        self.connect_calls: list[dict] = []
        self.pause_calls: list[dict] = []
        self.kill_calls: list[dict] = []
        self.set_timeout_calls: list[tuple[int, dict]] = []
        self.reconnect_effect: Exception | None = None
        self.pause_effect: Exception | None = None
        self.kill_effect: Exception | None = None
        self.set_timeout_effect: Exception | None = None
        self.pause_result = True
        self.kill_result = True

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        if self.reconnect_effect:
            raise self.reconnect_effect
        return self

    def set_timeout(self, timeout: int, **kwargs):
        self.set_timeout_calls.append((timeout, kwargs))
        if self.set_timeout_effect:
            raise self.set_timeout_effect

    def pause(self, **kwargs):
        self.pause_calls.append(kwargs)
        if self.pause_effect:
            raise self.pause_effect
        return self.pause_result

    def kill(self, **kwargs):
        self.kill_calls.append(kwargs)
        if self.kill_effect:
            raise self.kill_effect
        return self.kill_result


class SandboxService:
    def __init__(self):
        self.create_calls: list[dict] = []
        self.connect_calls: list[tuple[str, dict]] = []
        self.sessions: dict[str, SandboxSession] = {}
        self.connect_effect: Exception | None = None

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        session = SandboxSession(f"sb-{len(self.create_calls)}")
        self.sessions[session.sandbox_id] = session
        return session

    def connect(self, sandbox_id: str, **kwargs):
        self.connect_calls.append((sandbox_id, kwargs))
        if self.connect_effect:
            raise self.connect_effect
        return self.sessions.setdefault(sandbox_id, SandboxSession(sandbox_id))


class RecordingSyncManager:
    instances: list["RecordingSyncManager"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.sync_calls: list[bool] = []
        self.sync_back_calls: list[bool] = []
        self.load_state_calls: list[object] = []
        self.reset_calls = 0
        self.instances.append(self)

    def sync(self, *, force=False):
        self.sync_calls.append(force)
        return True

    def sync_back(self, hermes_home=None, *, require_prior_sync=True, restrict_to_roots=False):
        self.sync_back_calls.append((require_prior_sync, restrict_to_roots))
        return True

    def load_state(self, state, *, pending_state=None):
        self.load_state_calls.append((state, pending_state))
        return False

    def reset_remote_state(self):
        self.reset_calls += 1


class CleanupRecorder:
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


@pytest.fixture()
def e2b_backend(monkeypatch):
    service = SandboxService()

    e2b_root = types.ModuleType("e2b")
    setattr(
        e2b_root,
        "Sandbox",
        SimpleNamespace(create=service.create, connect=service.connect),
    )
    exceptions = types.ModuleType("e2b.exceptions")
    setattr(exceptions, "SandboxNotFoundException", SandboxMissing)
    setattr(exceptions, "RateLimitException", ApiRateLimited)
    setattr(exceptions, "FileNotFoundException", RemoteFileMissing)
    setattr(exceptions, "SandboxException", SandboxFailure)
    setattr(exceptions, "AuthenticationException", AuthenticationFailed)
    command_handle = types.ModuleType("e2b.sandbox.commands.command_handle")
    setattr(command_handle, "CommandExitException", CommandFailed)

    monkeypatch.setitem(sys.modules, "e2b", e2b_root)
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exceptions)
    monkeypatch.setitem(sys.modules, "e2b.sandbox", types.ModuleType("e2b.sandbox"))
    monkeypatch.setitem(
        sys.modules,
        "e2b.sandbox.commands",
        types.ModuleType("e2b.sandbox.commands"),
    )
    monkeypatch.setitem(
        sys.modules,
        "e2b.sandbox.commands.command_handle",
        command_handle,
    )

    from tools.environments import e2b as backend

    RecordingSyncManager.instances.clear()
    backend._ACTIVE_COMMANDS_BY_SANDBOX.clear()
    monkeypatch.setattr(backend, "_ensure_e2b_sdk", lambda: None)
    monkeypatch.setattr(backend, "FileSyncManager", RecordingSyncManager)
    monkeypatch.setattr(backend.E2BEnvironment, "init_session", lambda self: None)
    return backend, service


def make_environment(backend, **kwargs):
    return backend.E2BEnvironment(
        api_key="profile-key",
        task_id="task-1",
        timeout=30,
        lifetime_seconds=90,
        **kwargs,
    )


def test_persistent_create_resume_and_pause_contract(e2b_backend):
    backend, service = e2b_backend
    env = make_environment(backend, persistent_filesystem=True)

    create = service.create_calls[0]
    assert create["template"] == "base"
    assert create["timeout"] == 90
    assert create["api_key"] == "profile-key"
    assert create["lifecycle"] == {
        "on_timeout": {"action": "pause", "keep_memory": False},
        "auto_resume": False,
    }
    sandbox_id = env._sandbox_id
    assert service.sessions[sandbox_id].commands.calls[0][0] == (
        "mkdir -p /home/user/.hermes"
    )
    env.cleanup()
    assert service.sessions[sandbox_id].pause_calls == [
        {"keep_memory": False, "api_key": "profile-key"}
    ]

    resumed = make_environment(backend, persistent_filesystem=True)
    assert service.connect_calls == [
        (sandbox_id, {"timeout": 90, "api_key": "profile-key"})
    ]
    resumed.cleanup()


def test_ephemeral_cleanup_kills_sandbox(e2b_backend):
    backend, service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox
    assert service.create_calls[0]["lifecycle"] == {
        "on_timeout": "kill",
        "auto_resume": False,
    }

    env.cleanup()

    assert session.kill_calls == [{"api_key": "profile-key"}]
    assert session.pause_calls == []


def test_cleanup_does_not_interrupt_an_active_shared_command(e2b_backend, caplog):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=True)
    session = env._sandbox
    env._active_commands = 1
    caplog.set_level("INFO", logger=backend.__name__)

    env.cleanup()

    assert "deferring cleanup" in caplog.text
    assert session.pause_calls == []
    assert session.kill_calls == []
    assert env._sandbox is session

    env._active_commands = 0
    env.cleanup()
    assert session.pause_calls == [
        {"keep_memory": False, "api_key": "profile-key"}
    ]


def test_false_kill_result_means_sandbox_is_already_gone(e2b_backend, caplog):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox
    session.kill_result = False
    caplog.set_level("INFO", logger=backend.__name__)

    env.cleanup()

    assert "already gone" in caplog.text
    assert env._sandbox is None
    assert env._sandbox_id is None


def test_failed_ephemeral_kill_remains_retryable(e2b_backend, caplog):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox
    session.kill_effect = RuntimeError("kill unavailable")

    env.cleanup()

    assert "kill unavailable" in caplog.text
    assert env._sandbox is session
    assert env._sandbox_id == session.sandbox_id

    session.kill_effect = None
    env.cleanup()
    assert session.kill_calls == [
        {"api_key": "profile-key"},
        {"api_key": "profile-key"},
    ]
    assert env._sandbox is None
    assert env._sandbox_id is None


def test_missing_saved_sandbox_creates_fresh_but_transient_resume_does_not(
    e2b_backend,
):
    backend, service = e2b_backend
    backend._store_sandbox_record("task-1", "sb-stale", "base")
    service.connect_effect = SandboxMissing("gone")

    env = make_environment(backend, persistent_filesystem=True)
    assert env._sandbox_id == "sb-1"
    assert len(service.create_calls) == 1

    env.cleanup()
    backend._store_sandbox_record("task-1", "sb-rate-limited", "base")
    service.connect_effect = ApiRateLimited("try later")
    service.create_calls.clear()

    with pytest.raises(backend.EnvironmentConnectionError, match="try later"):
        make_environment(backend, persistent_filesystem=True)

    assert service.create_calls == []
    assert backend._load_sandbox_record("task-1", "base")["sandbox_id"] == "sb-rate-limited"


def test_failed_pause_preserves_persistence_pointer(e2b_backend, caplog):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=True)
    sandbox_id = env._sandbox_id
    env._sandbox.pause_effect = RuntimeError("pause unavailable")

    env.cleanup()

    assert "pause unavailable" in caplog.text
    assert backend._load_sandbox_record("task-1", "base")["sandbox_id"] == sandbox_id


def test_false_pause_result_preserves_persistence_pointer(e2b_backend, caplog):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=True)
    sandbox_id = env._sandbox_id
    env._sandbox.pause_result = False
    caplog.set_level("INFO", logger=backend.__name__)

    env.cleanup()

    assert "already paused" in caplog.text
    assert backend._load_sandbox_record("task-1", "base")["sandbox_id"] == sandbox_id


def test_command_cancellation_is_pid_scoped_and_race_safe(e2b_backend):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox
    run_entered = threading.Event()
    allow_handle = threading.Event()
    command_killed = threading.Event()

    class BlockingCommandHandle(RecordingCommandHandle):
        def wait(self, on_stdout=None, on_stderr=None):
            assert command_killed.wait(timeout=2)
            raise CommandFailed(stderr="killed", exit_code=137)

        def kill(self):
            super().kill()
            command_killed.set()
            return True

    handle = BlockingCommandHandle()

    def delayed_run(_command, kwargs):
        assert kwargs["background"] is True
        run_entered.set()
        assert allow_handle.wait(timeout=2)
        return handle

    session.commands.run_hook = delayed_run
    process = env._run_bash("sleep 30", timeout=1)
    assert run_entered.wait(timeout=2)
    assert env._active_commands == 1

    # A sibling gateway session may close while this shared environment is
    # executing. Cleanup must not pause/kill the sandbox under the command.
    env.cleanup()
    assert session.pause_calls == []
    assert session.kill_calls == []
    assert env._sandbox is session

    process.kill()
    allow_handle.set()

    assert process.wait(timeout=2) == 137
    assert env._active_commands == 0
    assert handle.kill_calls == 1
    assert session.kill_calls == []
    env.cleanup()


def test_sibling_cleanup_does_not_interrupt_shared_sandbox_command(e2b_backend):
    backend, _service = e2b_backend
    running_env = make_environment(backend, persistent_filesystem=True)
    sibling_env = make_environment(backend, persistent_filesystem=True)
    session = running_env._sandbox
    assert sibling_env._sandbox is session
    run_entered = threading.Event()
    command_killed = threading.Event()

    class BlockingCommandHandle(RecordingCommandHandle):
        def wait(self, on_stdout=None, on_stderr=None):
            run_entered.set()
            assert command_killed.wait(timeout=2)
            raise CommandFailed(stderr="killed", exit_code=137)

        def kill(self):
            super().kill()
            command_killed.set()
            return True

    handle = BlockingCommandHandle()
    session.commands.run_hook = lambda _command, _kwargs: handle
    process = running_env._run_bash("sleep 30", timeout=1)
    assert run_entered.wait(timeout=2)

    sibling_env.cleanup()
    assert session.pause_calls == []
    assert sibling_env._sandbox is session

    process.kill()
    assert process.wait(timeout=2) == 137
    assert backend._active_command_count(session.sandbox_id) == 0
    sibling_env.cleanup()
    assert session.pause_calls == [
        {"keep_memory": False, "api_key": "profile-key"}
    ]


def test_command_cancellation_after_handle_creation_kills_only_command(e2b_backend):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox
    wait_entered = threading.Event()
    command_killed = threading.Event()

    class WaitingCommandHandle(RecordingCommandHandle):
        def wait(self, on_stdout=None, on_stderr=None):
            wait_entered.set()
            assert command_killed.wait(timeout=2)
            raise CommandFailed(stderr="killed", exit_code=137)

        def kill(self):
            super().kill()
            command_killed.set()
            return True

    handle = WaitingCommandHandle()
    session.commands.run_hook = lambda _command, _kwargs: handle
    process = env._run_bash("sleep 30", timeout=1)
    assert wait_entered.wait(timeout=2)

    process.kill()

    assert process.wait(timeout=2) == 137
    assert handle.kill_calls == 1
    assert session.kill_calls == []
    env.cleanup()


def test_command_transport_failure_is_reported_as_backend_degradation(e2b_backend):
    backend, service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox

    def unavailable(_command, _kwargs):
        raise ApiRateLimited("capacity unavailable")

    env._sandbox.commands.run_hook = unavailable

    with pytest.raises(backend.EnvironmentConnectionError, match="capacity unavailable"):
        env.execute("echo hello", timeout=2)

    env.cleanup()
    assert session.kill_calls == [{"api_key": "profile-key"}]
    assert service.sessions[session.sandbox_id] is session


def test_cached_environment_renews_lease_for_longer_command_deadline(e2b_backend):
    """A later command with a longer timeout must extend the sandbox lease.

    The environment is created with ``lifetime_seconds=90``; without renewal
    the configured on-timeout lifecycle could pause the sandbox halfway
    through a valid 600-second command.
    """
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=True)
    session = env._sandbox

    short = env._run_bash("true", timeout=30)
    assert short.wait(timeout=2) == 0
    assert all(timeout >= 90 for timeout, _ in session.set_timeout_calls)

    long_cmd = env._run_bash("sleep 600", timeout=600)
    assert long_cmd.wait(timeout=2) == 0
    renewed, kwargs = session.set_timeout_calls[-1]
    assert renewed >= 605
    assert kwargs == {"api_key": "profile-key"}

    # A later short command must never truncate the longer lease.
    calls_after_long = list(session.set_timeout_calls)
    short_again = env._run_bash("true", timeout=30)
    assert short_again.wait(timeout=2) == 0
    assert session.set_timeout_calls == calls_after_long
    env.cleanup()


def test_failed_lease_renewal_surfaces_as_backend_degradation(e2b_backend):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    env._sandbox.set_timeout_effect = ApiRateLimited("throttled")

    with pytest.raises(backend.EnvironmentConnectionError, match="lease renewal"):
        env._run_bash("sleep 600", timeout=600)
    env._sandbox.set_timeout_effect = None
    env.cleanup()


def test_verbose_command_output_is_streamed_and_bounded(e2b_backend):
    """Output must be bounded while produced, not after full accumulation.

    Both retention points are checked: the Hermes-side collector keeps a
    head/tail window under the tool-output cap, and the SDK handle's private
    chunk buffers are drained as chunks are forwarded.
    """
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    session = env._sandbox

    chunk = "x" * 1024
    total_chunks = 500

    class SdkBufferingHandle(RecordingCommandHandle):
        def __init__(self):
            super().__init__()
            self._stdout_chunks: list[str] = []
            self._stderr_chunks: list[str] = []
            self.max_buffered = 0

        def wait(self, on_stdout=None, on_stderr=None):
            for _ in range(total_chunks):
                self._stdout_chunks.append(chunk)
                on_stdout(chunk)
                self.max_buffered = max(self.max_buffered, len(self._stdout_chunks))
            return CommandResult(exit_code=0)

    handle = SdkBufferingHandle()
    session.commands.run_hook = lambda _command, _kwargs: handle

    result = env.execute("spam", timeout=10, bounded_capture=True)

    assert result["returncode"] == 0
    assert "x" in result["output"]
    # Hermes-side retention stays within the tool-output window instead of
    # holding all 512k chars.
    assert len(result["output"]) < total_chunks * len(chunk) // 2
    # SDK-side buffers were drained during production, not at the end.
    assert handle.max_buffered <= 1
    assert handle._stdout_chunks == []
    env.cleanup()


def test_threaded_process_handle_requires_exactly_one_exec_fn():
    from tools.environments.base import _ThreadedProcessHandle

    with pytest.raises(ValueError, match="exactly one"):
        _ThreadedProcessHandle()
    with pytest.raises(ValueError, match="exactly one"):
        _ThreadedProcessHandle(
            lambda: ("", 0), stream_exec_fn=lambda write: 0
        )


def test_e2b_file_transport_uses_bulk_bytes_and_idempotent_delete(
    e2b_backend,
    tmp_path,
):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    first = tmp_path / "first.txt"
    second = tmp_path / "second.bin"
    first.write_text("hello", encoding="utf-8")
    second.write_bytes(b"\x00\x01")

    env._e2b_bulk_upload(
        [(str(first), "/home/user/.hermes/first.txt"), (str(second), "/tmp/second.bin")]
    )
    payload = env._sandbox.files.write_files_calls[0]
    assert payload == [
        {"path": "/home/user/.hermes/first.txt", "data": b"hello"},
        {"path": "/tmp/second.bin", "data": b"\x00\x01"},
    ]

    env._sandbox.files.remove_effects["/already-gone"] = RemoteFileMissing()
    env._e2b_delete(["/already-gone", "/present"])
    assert env._sandbox.files.remove_calls == ["/already-gone", "/present"]

    destination = tmp_path / "sync.tar"
    env._sandbox.files.read_payload = b"tar-bytes"
    env._e2b_bulk_download(destination)
    assert destination.read_bytes() == b"tar-bytes"
    assert env._sandbox.files.stream_readers[-1].closed


def test_bulk_download_stops_transferring_beyond_size_cap(
    e2b_backend,
    monkeypatch,
    tmp_path,
):
    backend, _service = e2b_backend
    env = make_environment(backend, persistent_filesystem=False)
    monkeypatch.setattr(backend, "_SYNC_BACK_MAX_BYTES", 10)
    env._sandbox.files.read_payload = b"0123456789abcdefghij"

    destination = tmp_path / "sync.tar"
    env._e2b_bulk_download(destination)

    transferred = destination.stat().st_size
    # The counting writer stops after crossing the cap instead of
    # materializing the whole archive; the partial file stays over the cap so
    # the sync-back size check refuses to extract it.
    assert 10 < transferred < 20
    assert env._sandbox.files.stream_readers[-1].closed
    env.cleanup()


def test_config_bridge_and_profile_scoped_cache_keys(monkeypatch):
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from tools import terminal_tool

    home = Path(__import__("os").environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "terminal:\n  backend: e2b\n  e2b_template: team-template\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_E2B_TEMPLATE", raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)

    config = terminal_tool._get_env_config()
    assert config["env_type"] == "e2b"
    assert config["e2b_template"] == "team-template"
    assert config["cwd"] == "/home/user"

    session_tokens = set_session_vars(session_key="gateway-session")
    try:
        token_a = set_hermes_home_override(home / "profile-a")
        try:
            key_a = terminal_tool._resolve_container_task_id(None)
        finally:
            reset_hermes_home_override(token_a)
        token_b = set_hermes_home_override(home / "profile-b")
        try:
            key_b = terminal_tool._resolve_container_task_id(None)
        finally:
            reset_hermes_home_override(token_b)
    finally:
        clear_session_vars(session_tokens)
        reset_session_vars()
    assert key_a != key_b
    assert key_a.endswith(":default")
    assert key_b.endswith(":default")


def test_e2b_cache_key_is_stable_across_gateway_sessions(monkeypatch):
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from hermes_constants import hermes_home_key
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "e2b")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "true")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    expected = f"e2b:{hermes_home_key()}:default"
    resolved = []

    try:
        for session_key in ("gateway-session-a", "gateway-session-b"):
            tokens = set_session_vars(session_key=session_key)
            try:
                parent_key = terminal_tool._resolve_container_task_id(None)
                child_key = terminal_tool._resolve_container_task_id("subagent-task")
            finally:
                clear_session_vars(tokens)
            resolved.append(parent_key)
            assert child_key == parent_key
    finally:
        reset_session_vars()

    assert resolved == [expected, expected]


def test_nonpersistent_e2b_isolates_sessions_but_subagents_share_parent(monkeypatch):
    """Paired with the persistent test above: with persistence disabled, two
    gateway sessions must NOT share one nominally ephemeral sandbox, while a
    subagent still shares its parent session's sandbox."""
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from hermes_constants import hermes_home_key
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "e2b")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    resolved: dict[str, str] = {}

    try:
        for session_key in ("gateway-session-a", "gateway-session-b"):
            tokens = set_session_vars(session_key=session_key)
            try:
                parent_key = terminal_tool._resolve_container_task_id(None)
                child_key = terminal_tool._resolve_container_task_id("subagent-task")
            finally:
                clear_session_vars(tokens)
            assert child_key == parent_key
            resolved[session_key] = parent_key
    finally:
        reset_session_vars()

    prefix = f"e2b:{hermes_home_key()}:"
    assert resolved["gateway-session-a"] == f"{prefix}session:gateway-session-a"
    assert resolved["gateway-session-b"] == f"{prefix}session:gateway-session-b"

    # Outside a session (CLI), non-persistent E2B keeps the shared default
    # slot — single-session processes are unaffected.
    assert terminal_tool._resolve_container_task_id(None) == f"{prefix}default"
    assert terminal_tool._resolve_container_task_id("subagent-task") == (
        f"{prefix}default"
    )


def test_e2b_isolation_override_still_wins_in_nonpersistent_mode(monkeypatch):
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from hermes_constants import hermes_home_key
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "e2b")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    terminal_tool.register_task_env_overrides("benchmark-task", {"env_type": "e2b"})
    tokens = set_session_vars(session_key="gateway-session")
    try:
        assert terminal_tool._resolve_container_task_id("benchmark-task") == (
            f"e2b:{hermes_home_key()}:benchmark-task"
        )
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
        terminal_tool.clear_task_env_overrides("benchmark-task")


def test_e2b_isolation_override_wins_without_losing_profile_scope(monkeypatch):
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from hermes_constants import hermes_home_key
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "e2b")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    terminal_tool.register_task_env_overrides("benchmark-task", {"env_type": "e2b"})
    tokens = set_session_vars(session_key="gateway-session")
    try:
        assert terminal_tool._resolve_container_task_id("benchmark-task") == (
            f"e2b:{hermes_home_key()}:benchmark-task"
        )
    finally:
        clear_session_vars(tokens)
        reset_session_vars()
        terminal_tool.clear_task_env_overrides("benchmark-task")


def test_gateway_sessions_reuse_e2b_environment_and_cleanup_after_context_clears(
    monkeypatch,
):
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "e2b")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    env = CleanupRecorder()
    created_task_ids = []

    def create_environment(**kwargs):
        created_task_ids.append(kwargs["task_id"])
        return env

    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)

    try:
        for session_key in ("gateway-session-a", "gateway-session-b"):
            tokens = set_session_vars(session_key=session_key)
            try:
                assert terminal_tool.ensure_task_env(None) is env
            finally:
                clear_session_vars(tokens)
    finally:
        reset_session_vars()

    scoped_key = terminal_tool._resolve_container_task_id(None)
    assert created_task_ids == [scoped_key]
    assert terminal_tool._active_environments == {scoped_key: env}

    terminal_tool.cleanup_vm("gateway-session-b")

    assert env.cleanup_calls == 1
    assert terminal_tool._active_environments == {}
    assert terminal_tool._last_activity == {}


def test_nonpersistent_gateway_sessions_get_separate_environments(monkeypatch):
    """Production-path pair to the persistent reuse test above: with
    persistence disabled, two sessions create two environments, and one
    session's cleanup must not tear down the other's."""
    from gateway.session_context import (
        clear_session_vars,
        reset_session_vars,
        set_session_vars,
    )
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "e2b")
    monkeypatch.setenv("TERMINAL_CONTAINER_PERSISTENT", "false")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)
    envs_by_task: dict[str, CleanupRecorder] = {}

    def create_environment(**kwargs):
        env = CleanupRecorder()
        envs_by_task[kwargs["task_id"]] = env
        return env

    monkeypatch.setattr(terminal_tool, "_active_environments", {})
    monkeypatch.setattr(terminal_tool, "_last_activity", {})
    monkeypatch.setattr(terminal_tool, "_creation_locks", {})
    monkeypatch.setattr(terminal_tool, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal_tool, "_create_environment", create_environment)

    session_envs: dict[str, CleanupRecorder] = {}
    try:
        for session_key in ("gateway-session-a", "gateway-session-b"):
            tokens = set_session_vars(session_key=session_key)
            try:
                session_envs[session_key] = terminal_tool.ensure_task_env(None)
            finally:
                clear_session_vars(tokens)

        assert len(envs_by_task) == 2
        assert session_envs["gateway-session-a"] is not session_envs["gateway-session-b"]

        # Session B closes while its context is still active — only B's
        # sandbox may be cleaned.
        tokens = set_session_vars(session_key="gateway-session-b")
        try:
            terminal_tool.cleanup_vm("gateway-session-b")
        finally:
            clear_session_vars(tokens)
    finally:
        reset_session_vars()

    assert session_envs["gateway-session-b"].cleanup_calls == 1
    assert session_envs["gateway-session-a"].cleanup_calls == 0
    assert list(terminal_tool._active_environments.values()) == [
        session_envs["gateway-session-a"]
    ]


def test_cleanup_vm_resolves_config_only_e2b_key(monkeypatch):
    from hermes_constants import hermes_home_key
    from tools import terminal_tool

    home = Path(__import__("os").environ["HERMES_HOME"])
    (home / "config.yaml").write_text(
        "terminal:\n  backend: e2b\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", False)

    env = CleanupRecorder()
    scoped_key = f"e2b:{hermes_home_key()}:default"
    monkeypatch.setattr(terminal_tool, "_active_environments", {scoped_key: env})
    monkeypatch.setattr(terminal_tool, "_last_activity", {scoped_key: 1.0})

    terminal_tool.cleanup_vm("public-task")

    assert env.cleanup_calls == 1
    assert terminal_tool._active_environments == {}
    assert terminal_tool._last_activity == {}


def test_cleanup_vm_does_not_drop_legacy_raw_e2b_environment(monkeypatch):
    from tools import terminal_tool

    monkeypatch.setenv("TERMINAL_ENV", "e2b")
    monkeypatch.setattr(terminal_tool, "_terminal_config_bridge_attempted", True)

    scoped = CleanupRecorder()
    legacy = CleanupRecorder()
    scoped_key = terminal_tool._resolve_container_task_id("public-task")
    monkeypatch.setattr(
        terminal_tool,
        "_active_environments",
        {scoped_key: scoped, "public-task": legacy},
    )
    monkeypatch.setattr(
        terminal_tool,
        "_last_activity",
        {scoped_key: 1.0, "public-task": 1.0},
    )

    terminal_tool.cleanup_vm("public-task")

    assert scoped.cleanup_calls == 1
    assert legacy.cleanup_calls == 1
    assert terminal_tool._active_environments == {}
    assert terminal_tool._last_activity == {}


def test_factory_uses_scoped_key_and_never_foreign_process_secret(
    e2b_backend,
    monkeypatch,
):
    _backend, service = e2b_backend
    from agent import secret_scope
    from tools import terminal_tool
    from tools.environments.local import _HERMES_PROVIDER_ENV_BLOCKLIST

    assert "E2B_API_KEY" in _HERMES_PROVIDER_ENV_BLOCKLIST
    monkeypatch.setenv("E2B_API_KEY", "foreign-process-key")
    secret_scope.set_multiplex_active(True)
    token = secret_scope.set_secret_scope({"E2B_API_KEY": "profile-a-key"})
    try:
        env = terminal_tool._create_environment(
            env_type="e2b",
            image="",
            cwd="/home/user",
            timeout=30,
            container_config={
                "e2b_template": "base",
                "lifetime_seconds": 90,
                "container_persistent": False,
            },
            task_id="profile-a",
        )
        assert service.create_calls[-1]["api_key"] == "profile-a-key"
        env.cleanup()
    finally:
        secret_scope.reset_secret_scope(token)

    empty = secret_scope.set_secret_scope({})
    try:
        with pytest.raises(ValueError, match="E2B_API_KEY"):
            terminal_tool._create_environment(
                env_type="e2b",
                image="",
                cwd="/home/user",
                timeout=30,
                container_config={"container_persistent": False},
                task_id="profile-b",
            )
    finally:
        secret_scope.reset_secret_scope(empty)
        secret_scope.set_multiplex_active(False)


def test_persistence_store_is_structured_and_template_scoped(e2b_backend):
    backend, _service = e2b_backend
    backend._store_sandbox_record("task-1", "sb-base", "base")
    backend._store_sandbox_record("task-1", "sb-custom", "custom")

    assert backend._load_sandbox_record("task-1", "base")["sandbox_id"] == "sb-base"
    assert backend._load_sandbox_record("task-1", "custom")["sandbox_id"] == "sb-custom"
    raw = json.loads(backend._sandbox_store_path().read_text(encoding="utf-8"))
    assert len(raw) == 2


def test_pending_sync_state_is_atomically_promoted(e2b_backend):
    backend, _service = e2b_backend
    backend._store_sandbox_record("task-1", "sb-base", "base")
    pending = {
        "synced_files": {"/remote": [1.0, 2]},
        "pushed_hashes": {"/remote": "pending-hash"},
        "upload_only_host_paths": [],
    }
    committed = {
        "synced_files": {"/remote": [2.0, 3]},
        "pushed_hashes": {"/remote": "committed-hash"},
        "upload_only_host_paths": [],
    }

    backend._store_sandbox_sync_state(
        "task-1",
        "base",
        "sb-base",
        pending,
        pending=True,
    )
    assert backend._load_sandbox_pending_sync_state(
        "task-1", "base", "sb-base"
    ) == pending
    assert backend._load_sandbox_sync_state("task-1", "base", "sb-base") is None

    backend._store_sandbox_sync_state(
        "task-1",
        "base",
        "sb-base",
        committed,
    )
    assert backend._load_sandbox_sync_state(
        "task-1", "base", "sb-base"
    ) == committed
    assert (
        backend._load_sandbox_pending_sync_state("task-1", "base", "sb-base")
        is None
    )


def test_persistence_store_preserves_symlink_target(e2b_backend):
    backend, _service = e2b_backend
    store = backend._sandbox_store_path()
    target = store.with_name("sandbox-store-target.json")
    target.write_text("{}", encoding="utf-8")
    try:
        store.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    backend._store_sandbox_record("task-1", "sb-base", "base")

    assert store.is_symlink()
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw[backend._record_key("task-1", "base")]["sandbox_id"] == "sb-base"


def test_setup_wizard_persists_e2b_backend_template_and_key(monkeypatch):
    from hermes_cli import setup

    saved_env: dict[str, str] = {}
    monkeypatch.setattr(setup, "prompt_choice", lambda *_args: 5)
    monkeypatch.setattr(setup, "get_env_value", lambda _name: None)
    monkeypatch.setattr(
        setup,
        "prompt",
        lambda _label, default=None, password=False: (
            "profile-api-key" if password else "team-template"
        ),
    )
    monkeypatch.setattr(setup, "prompt_yes_no", lambda *_args: True)
    monkeypatch.setattr(setup, "save_env_value", saved_env.__setitem__)
    monkeypatch.setattr(setup, "save_config", lambda _config: None)
    monkeypatch.setattr(setup, "print_header", lambda *_args: None)
    monkeypatch.setattr(setup, "print_info", lambda *_args: None)
    monkeypatch.setattr(setup, "print_success", lambda *_args: None)

    config = {"terminal": {"backend": "local"}}
    setup.setup_terminal_backend(config)

    assert config["terminal"] == {
        "backend": "e2b",
        "e2b_template": "team-template",
        "container_persistent": True,
    }
    assert saved_env == {
        "E2B_API_KEY": "profile-api-key",
        "TERMINAL_ENV": "e2b",
        "TERMINAL_E2B_TEMPLATE": "team-template",
    }
