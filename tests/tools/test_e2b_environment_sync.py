"""Regression tests for the E2B backend's file-sync contract.

These tests deliberately use the real :class:`FileSyncManager`.  The remote
filesystem double rejects writes to missing parent directories even though the
current E2B service creates them, so Hermes owns that portability guarantee.
"""

from __future__ import annotations

import io
import posixpath
import shlex
import sys
import tarfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_constants import get_hermes_home
from tools.environments import e2b as backend
from tools.environments.file_sync import FileSyncManager, iter_sync_files


class SandboxMissing(Exception):
    pass


class RemoteFileMissing(Exception):
    pass


class StrictRemoteFiles:
    """Minimal E2B file API that requires parents to exist before writes."""

    def __init__(self):
        self.directories = {"/", "/home", "/home/user"}
        self.files: dict[str, bytes] = {}
        self.fail_writes = False
        self.download_payload = _tar_bytes({})

    def mkdir_p(self, path: str) -> None:
        current = ""
        for part in path.strip("/").split("/"):
            current += f"/{part}"
            self.directories.add(current)

    def write(self, path: str, data: bytes) -> None:
        self.write_files([{"path": path, "data": data}])

    def write_files(self, payload: list[dict]) -> None:
        if self.fail_writes:
            raise RuntimeError("E2B upload unavailable")
        for entry in payload:
            parent = posixpath.dirname(entry["path"])
            if parent not in self.directories:
                raise RuntimeError(f"missing remote directory: {parent}")
        for entry in payload:
            self.files[entry["path"]] = entry["data"]

    def read(self, _path: str, *, format: str):
        assert format == "stream"
        return io.BytesIO(bytes(self.download_payload))

    def remove(self, path: str) -> None:
        self.files.pop(path, None)


class RemoteCommands:
    def __init__(self, files: StrictRemoteFiles):
        self.files = files
        self.calls: list[tuple[str, dict]] = []
        self.fail_tar = False

    def run(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        parts = shlex.split(command)
        if parts[:2] == ["mkdir", "-p"]:
            for path in parts[2:]:
                self.files.mkdir_p(path)
        elif parts[:2] == ["tar", "cf"]:
            # Regenerate the download archive from the live remote files so a
            # sync-back observes exactly what the sandbox holds right now.
            if self.fail_tar:
                raise RuntimeError("tar transport unavailable")
            self.files.download_payload = _tar_bytes(
                {
                    path.lstrip("/"): data
                    for path, data in self.files.files.items()
                }
            )
        return SimpleNamespace(stdout="", stderr="", exit_code=0)


class SandboxSession:
    def __init__(self, sandbox_id: str, *, fail_writes: bool = False):
        self.sandbox_id = sandbox_id
        self.files = StrictRemoteFiles()
        self.files.fail_writes = fail_writes
        self.commands = RemoteCommands(self.files)
        self.connect_calls: list[dict] = []
        self.pause_calls: list[dict] = []
        self.kill_calls: list[dict] = []
        self.reconnect_effect: Exception | None = None

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        if self.reconnect_effect is not None:
            raise self.reconnect_effect
        return self

    def pause(self, **kwargs):
        self.pause_calls.append(kwargs)
        return True

    def kill(self, **kwargs):
        self.kill_calls.append(kwargs)
        return True

    def set_timeout(self, timeout: int, **kwargs):
        return None


class SandboxService:
    def __init__(self):
        self.sessions: dict[str, SandboxSession] = {}
        self.fail_writes = False

    def create(self, **_kwargs):
        session = SandboxSession(
            f"sb-{len(self.sessions) + 1}",
            fail_writes=self.fail_writes,
        )
        self.sessions[session.sandbox_id] = session
        return session

    def connect(self, sandbox_id: str, **_kwargs):
        if sandbox_id not in self.sessions:
            raise SandboxMissing(sandbox_id)
        return self.sessions[sandbox_id]


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, content in files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture()
def e2b_service(monkeypatch):
    service = SandboxService()

    e2b_module = types.ModuleType("e2b")
    e2b_module.Sandbox = SimpleNamespace(
        create=service.create,
        connect=service.connect,
    )
    exceptions = types.ModuleType("e2b.exceptions")
    exceptions.SandboxNotFoundException = SandboxMissing
    exceptions.FileNotFoundException = RemoteFileMissing

    monkeypatch.setitem(sys.modules, "e2b", e2b_module)
    monkeypatch.setitem(sys.modules, "e2b.exceptions", exceptions)
    monkeypatch.setattr(backend, "_ensure_e2b_sdk", lambda: None)
    monkeypatch.setattr(backend.E2BEnvironment, "init_session", lambda self: None)
    return service


def _environment(**kwargs) -> backend.E2BEnvironment:
    return backend.E2BEnvironment(
        api_key="profile-key",
        task_id="sync-contract",
        timeout=30,
        lifetime_seconds=90,
        **kwargs,
    )


def test_real_sync_inventory_excludes_profile_env_and_config():
    hermes_home = get_hermes_home()
    profile_env = hermes_home / ".env"
    profile_config = hermes_home / "config.yaml"
    skill = hermes_home / "skills" / "incident-triage" / "SKILL.md"
    profile_env.write_text("E2B_API_KEY=must-not-upload\n", encoding="utf-8")
    profile_config.write_text("terminal:\n  backend: e2b\n", encoding="utf-8")
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text("triage", encoding="utf-8")

    inventory = iter_sync_files("/home/user/.hermes")
    host_paths = {host for host, _remote in inventory}

    assert (str(skill), "/home/user/.hermes/skills/incident-triage/SKILL.md") in inventory
    assert str(profile_env) not in host_paths
    assert str(profile_config) not in host_paths


def test_real_sync_manager_creates_nested_parents_before_upload(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    mappings = []
    for relative in (
        "skills/research/arxiv/SKILL.md",
        "skills/github/SKILL.md",
        "cache/models/catalog.json",
    ):
        host = tmp_path / relative
        host.parent.mkdir(parents=True, exist_ok=True)
        host.write_text(relative, encoding="utf-8")
        mappings.append((str(host), f"/home/user/.hermes/{relative}"))
    monkeypatch.setattr(backend, "iter_sync_files", lambda _base: mappings)

    env = _environment(persistent_filesystem=False)
    session = env._sandbox

    assert session.files.files == {
        remote: Path(host).read_bytes() for host, remote in mappings
    }
    env.cleanup()


def test_failed_initial_sync_fails_startup_and_cleans_up_new_sandbox(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    host = tmp_path / "skills" / "incident-triage" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("triage", encoding="utf-8")
    monkeypatch.setattr(
        backend,
        "iter_sync_files",
        lambda _base: [(str(host), "/home/user/.hermes/skills/incident-triage/SKILL.md")],
    )
    e2b_service.fail_writes = True

    with pytest.raises(backend.EnvironmentConnectionError, match="initial state sync"):
        _environment(persistent_filesystem=True)

    session = e2b_service.sessions["sb-1"]
    assert session.kill_calls == [{"api_key": "profile-key"}]
    assert backend._load_sandbox_record("sync-contract", "base") is None


def test_failed_initial_sync_preserves_resumed_sandbox(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    host = tmp_path / "skills" / "existing" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        backend,
        "iter_sync_files",
        lambda _base: [(str(host), "/home/user/.hermes/skills/existing/SKILL.md")],
    )

    first = _environment(persistent_filesystem=True)
    sandbox_id = first._sandbox_id
    session = first._sandbox
    first.cleanup()
    # A host edit forces upload work on resume; the loaded baseline would
    # otherwise correctly report the unchanged snapshot as already synced.
    host.write_text("existing-but-updated", encoding="utf-8")
    session.files.fail_writes = True

    with pytest.raises(backend.EnvironmentConnectionError, match="initial state sync"):
        _environment(persistent_filesystem=True)

    assert session.kill_calls == []
    assert session.pause_calls == [
        {"keep_memory": False, "api_key": "profile-key"},
        {"keep_memory": False, "api_key": "profile-key"},
    ]
    assert backend._load_sandbox_record("sync-contract", "base") == {
        "sandbox_id": sandbox_id,
        "template": "base",
    }


def test_successful_empty_initial_sync_pulls_new_skill_and_memory_directories(
    e2b_service,
    monkeypatch,
):
    monkeypatch.setattr(backend, "iter_sync_files", lambda _base: [])
    env = _environment(persistent_filesystem=True)
    env._sandbox.files.files.update(
        {
            "/home/user/.hermes/skills/incident-triage/SKILL.md": b"triage skill",
            "/home/user/.hermes/memories/MEMORY.md": b"incident lesson",
        }
    )

    env.cleanup()

    hermes_home = get_hermes_home()
    assert (hermes_home / "skills/incident-triage/SKILL.md").read_bytes() == b"triage skill"
    assert (hermes_home / "memories/MEMORY.md").read_bytes() == b"incident lesson"


def test_missing_active_sandbox_is_recreated_and_state_is_reuploaded(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    host = tmp_path / "skills" / "incident-triage" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("triage", encoding="utf-8")
    remote = "/home/user/.hermes/skills/incident-triage/SKILL.md"
    monkeypatch.setattr(
        backend,
        "iter_sync_files",
        lambda _base: [(str(host), remote)],
    )
    env = _environment(persistent_filesystem=True)
    stale = env._sandbox
    stale.reconnect_effect = SandboxMissing("removed outside Hermes")

    env._before_execute()

    replacement = env._sandbox
    assert replacement is not stale
    assert replacement.files.files[remote] == b"triage"
    assert backend._load_sandbox_record("sync-contract", "base")["sandbox_id"] == (
        replacement.sandbox_id
    )
    env.cleanup()


def test_cleanup_does_not_recreate_missing_sandbox(
    e2b_service,
    monkeypatch,
):
    monkeypatch.setattr(backend, "iter_sync_files", lambda _base: [])
    env = _environment(persistent_filesystem=True)
    stale = env._sandbox
    stale.reconnect_effect = SandboxMissing("removed outside Hermes")

    env.cleanup()

    assert len(e2b_service.sessions) == 1
    assert stale.pause_calls == []
    assert backend._load_sandbox_record("sync-contract", "base") is None


def _host_state_iter(tmp_path: Path):
    """iter_sync_files stand-in that reflects the live host tree each cycle."""

    def fake_iter(_base: str) -> list[tuple[str, str]]:
        return [
            (str(path), f"/home/user/.hermes/{path.relative_to(tmp_path).as_posix()}")
            for path in sorted(tmp_path.rglob("*"))
            if path.is_file()
        ]

    return fake_iter


def test_resume_recovers_remote_edit_after_crash_without_cleanup(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    """P1: a resumed sandbox's newer state must be pulled before any push.

    Reviewer reproduction — host skill contains ``old``, the persisted sandbox
    contains ``new``, the record is retained, cleanup never ran. Construction
    must recover ``new`` instead of writing ``old`` over it.
    """
    monkeypatch.setattr(backend, "iter_sync_files", _host_state_iter(tmp_path))
    host = tmp_path / "skills" / "incident-triage" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("old", encoding="utf-8")
    remote = "/home/user/.hermes/skills/incident-triage/SKILL.md"

    first = _environment(persistent_filesystem=True)
    session = first._sandbox
    assert session.files.files[remote] == b"old"
    # Agent-authored edit inside the sandbox, then process death: no cleanup,
    # no sync_back — only the record and its committed baseline survive.
    session.files.files[remote] = b"new"
    first._sandbox = None
    first._sandbox_id = None

    second = _environment(persistent_filesystem=True)

    assert host.read_bytes() == b"new"
    assert session.files.files[remote] == b"new"
    second.cleanup()


def test_resume_after_failed_sync_back_recovers_remote_edit(
    e2b_service,
    monkeypatch,
    tmp_path,
    caplog,
):
    """A sync-back that exhausts its retries must not promote the host copy.

    Cleanup pauses the sandbox but reports the un-pulled changes; the next
    construction recovers them before pushing the (stale) host snapshot.
    """
    monkeypatch.setattr(backend, "iter_sync_files", _host_state_iter(tmp_path))
    monkeypatch.setattr("tools.environments.file_sync._sleep", lambda _delay: None)
    host = tmp_path / "skills" / "incident-triage" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("old", encoding="utf-8")
    remote = "/home/user/.hermes/skills/incident-triage/SKILL.md"

    first = _environment(persistent_filesystem=True)
    session = first._sandbox
    session.files.files[remote] = b"new"
    session.commands.fail_tar = True

    first.cleanup()

    assert host.read_bytes() == b"old"
    assert session.pause_calls, "sandbox must still be preserved for recovery"
    assert "will be recovered on the next resume" in caplog.text

    session.commands.fail_tar = False
    second = _environment(persistent_filesystem=True)

    assert host.read_bytes() == b"new"
    assert session.files.files[remote] == b"new"
    second.cleanup()


def test_resume_without_baseline_recovers_remote_state_before_push(
    e2b_service,
    monkeypatch,
    caplog,
    tmp_path,
):
    """A record without a committed baseline still recovers remote state.

    Only the declared agent-state roots (skills, memories) may be written:
    without a baseline nothing distinguishes host-owned files from
    remote-authored ones, so a tampered remote credential must not be
    recreated on the host even when prefix inference could map it there.
    """
    monkeypatch.setattr(backend, "iter_sync_files", _host_state_iter(tmp_path))
    # A surviving host credential gives prefix inference a sibling to map
    # the tampered remote credential through — the exact channel that must
    # stay closed during baseline-less recovery.
    other_credential = tmp_path / "credentials" / "other.json"
    other_credential.parent.mkdir(parents=True)
    other_credential.write_text("kept", encoding="utf-8")
    remote = "/home/user/.hermes/skills/agent-made/SKILL.md"
    session = SandboxSession("sb-legacy")
    session.files.mkdir_p("/home/user/.hermes/skills/agent-made")
    session.files.mkdir_p("/home/user/.hermes/credentials")
    session.files.files[remote] = b"agent skill"
    session.files.files["/home/user/.hermes/credentials/token.json"] = b"tampered"
    e2b_service.sessions["sb-legacy"] = session
    backend._store_sandbox_record("sync-contract", "sb-legacy", "base")

    env = _environment(persistent_filesystem=True)

    recovered = get_hermes_home() / "skills" / "agent-made" / "SKILL.md"
    assert recovered.read_bytes() == b"agent skill"
    assert "no committed sync baseline" in caplog.text
    assert not (tmp_path / "credentials" / "token.json").exists()
    env.cleanup()


def test_host_deletion_while_paused_removes_remote_file_on_resume(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    """A file deleted on the host must not stay readable in the sandbox."""
    monkeypatch.setattr(backend, "iter_sync_files", _host_state_iter(tmp_path))
    stale = tmp_path / "credentials" / "token.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("secret", encoding="utf-8")
    keep = tmp_path / "skills" / "github" / "SKILL.md"
    keep.parent.mkdir(parents=True)
    keep.write_text("keep", encoding="utf-8")
    stale_remote = "/home/user/.hermes/credentials/token.json"
    keep_remote = "/home/user/.hermes/skills/github/SKILL.md"

    first = _environment(persistent_filesystem=True)
    session = first._sandbox
    assert session.files.files[stale_remote] == b"secret"
    first.cleanup()

    stale.unlink()
    second = _environment(persistent_filesystem=True)

    assert stale_remote not in session.files.files
    assert session.files.files[keep_remote] == b"keep"
    assert not stale.exists(), "recovery must not resurrect the deleted file"
    second.cleanup()


def test_failed_pre_command_sync_prevents_execution(
    e2b_service,
    monkeypatch,
    tmp_path,
):
    """A command must not run against stale state after a failed sync."""
    monkeypatch.setattr(backend, "iter_sync_files", _host_state_iter(tmp_path))
    host = tmp_path / "skills" / "incident-triage" / "SKILL.md"
    host.parent.mkdir(parents=True)
    host.write_text("v1", encoding="utf-8")

    env = _environment(persistent_filesystem=True)
    session = env._sandbox
    env._sync_manager._sync_interval = 0.0
    host.write_text("v2-with-longer-body", encoding="utf-8")
    session.files.fail_writes = True

    with pytest.raises(
        backend.EnvironmentConnectionError, match="state sync before command"
    ):
        env.execute("echo hello", timeout=5)

    assert all("echo hello" not in command for command, _ in session.commands.calls)
    session.files.fail_writes = False
    env.cleanup()


def test_sync_reports_transaction_failure_to_caller(tmp_path):
    host = tmp_path / "skill.md"
    host.write_text("state", encoding="utf-8")

    def fail_upload(_host: str, _remote: str) -> None:
        raise RuntimeError("offline")

    manager = FileSyncManager(
        get_files_fn=lambda: [(str(host), "/home/user/.hermes/skills/skill.md")],
        upload_fn=fail_upload,
        delete_fn=lambda _paths: None,
    )

    assert manager.sync(force=True) is False
