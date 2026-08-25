"""Shared file sync manager for remote execution backends.

Tracks local file changes via mtime+size, detects deletions, and syncs to
remote environments transactionally. Docker and Singularity use bind mounts
(live host FS view) and don't need this.
"""

import hashlib
import logging
import os
import posixpath
import shlex
import shutil
import signal
import sys
import tarfile
import tempfile
import threading
import time

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[invalid-assignment]  # Windows — locking skipped
from pathlib import Path, PurePosixPath
from typing import Callable, Mapping, cast

from hermes_constants import get_hermes_home
from tools.environments.base import _file_mtime_key

logger = logging.getLogger(__name__)

# Keep retry sleeps patchable without mutating the shared stdlib ``time``
# module. Patching ``tools.environments.file_sync.time.sleep`` replaces
# ``time.sleep`` globally because ``time`` is the module object; under xdist
# that lets unrelated background threads inflate retry-test call counts.
_sleep = time.sleep
# Same rationale for the rate-limit clock: tests patch ``_monotonic``
# instead of ``time.monotonic`` on the shared module object.
_monotonic = time.monotonic

_SYNC_INTERVAL_SECONDS = 5.0
_FORCE_SYNC_ENV = "HERMES_FORCE_FILE_SYNC"

# Transport callbacks provided by each backend
UploadFn = Callable[[str, str], None]  # (host_path, remote_path) -> raises on failure
BulkUploadFn = Callable[[list[tuple[str, str]]], None]  # [(host_path, remote_path), ...] -> raises on failure
BulkDownloadFn = Callable[[Path], None]  # (dest_tar_path) -> writes tar archive, raises on failure
DeleteFn = Callable[[list[str]], None]  # (remote_paths) -> raises on failure
GetFilesFn = Callable[[], list[tuple[str, str]]]  # () -> [(host_path, remote_path), ...]


def iter_sync_files(container_base: str = "/root/.hermes") -> list[tuple[str, str]]:
    """Enumerate all files that should be synced to a remote environment.

    Combines credentials, skills, and cache into a single flat list of
    (host_path, remote_path) pairs.  Credential paths are remapped from
    the hardcoded /root/.hermes to *container_base* because the remote
    user's home may differ (e.g. /home/daytona, /home/user).
    """
    # Late import: credential_files imports agent modules that create
    # circular dependencies if loaded at file_sync module level.
    from tools.credential_files import (
        get_credential_file_mounts,
        iter_cache_files,
        iter_skills_files,
    )

    files: list[tuple[str, str]] = []
    for entry in get_credential_file_mounts():
        remote = entry["container_path"].replace(
            "/root/.hermes", container_base, 1
        )
        files.append((entry["host_path"], remote))
    for entry in iter_skills_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    for entry in iter_cache_files(container_base=container_base):
        files.append((entry["host_path"], entry["container_path"]))
    return files


def _credential_host_paths() -> set[str]:
    """Return credential files that are upload-only for remote sandboxes."""
    try:
        from tools.credential_files import get_credential_file_mounts
    except Exception:
        return set()

    paths: set[str] = set()
    try:
        mounts = get_credential_file_mounts()
    except Exception:
        return set()
    for entry in mounts:
        host_path = entry.get("host_path") if isinstance(entry, dict) else None
        if not host_path:
            continue
        try:
            paths.add(str(Path(host_path).expanduser().resolve()))
        except OSError:
            paths.add(str(Path(host_path).expanduser()))
    return paths


def quoted_rm_command(remote_paths: list[str]) -> str:
    """Build a shell ``rm -f`` command for a batch of remote paths."""
    return "rm -f " + " ".join(shlex.quote(p) for p in remote_paths)


def quoted_mkdir_command(dirs: list[str]) -> str:
    """Build a shell ``mkdir -p`` command for a batch of directories."""
    return "mkdir -p " + " ".join(shlex.quote(d) for d in dirs)


def unique_parent_dirs(files: list[tuple[str, str]]) -> list[str]:
    """Extract sorted unique parent directories from (host, remote) pairs."""
    return sorted({posixpath.dirname(remote) for _, remote in files})


def _sha256_file(path: str) -> str:
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


_SYNC_BACK_MAX_RETRIES = 3
_SYNC_BACK_BACKOFF = (2, 4, 8)  # seconds between retries
_SYNC_BACK_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB — refuse to extract larger tars
_PENDING_OPERATION_KEY = "pending_operation"
_PENDING_PUSH = "push"
_PENDING_PULL = "pull"


class _PermanentSyncBackError(RuntimeError):
    """Non-transient sync-back rejection that retries cannot repair."""


class FileSyncManager:
    """Tracks local file changes and syncs to a remote environment.

    Backends instantiate this with transport callbacks (upload, delete)
    and a file-source callable.  The manager handles mtime-based change
    detection, deletion tracking, rate limiting, and transactional state.

    Not used by bind-mount backends (Docker, Singularity) — those get
    live host FS views and don't need file sync.
    """

    def __init__(
        self,
        get_files_fn: GetFilesFn,
        upload_fn: UploadFn,
        delete_fn: DeleteFn,
        sync_interval: float = _SYNC_INTERVAL_SECONDS,
        bulk_upload_fn: BulkUploadFn | None = None,
        bulk_download_fn: BulkDownloadFn | None = None,
        sync_back_roots: list[tuple[str, str]] | None = None,
        on_state_pending: Callable[[dict], None] | None = None,
        on_state_committed: Callable[[dict], None] | None = None,
    ):
        self._get_files_fn = get_files_fn
        self._upload_fn = upload_fn
        self._bulk_upload_fn = bulk_upload_fn
        self._bulk_download_fn = bulk_download_fn
        self._delete_fn = delete_fn
        self._sync_back_roots = [
            (Path(host).expanduser().resolve(), posixpath.normpath(remote))
            for host, remote in (sync_back_roots or [])
        ]
        # Cross-process backends use a write-ahead pending snapshot before any
        # remote or host mutation, then atomically promote the committed
        # snapshot after the transaction. A process death between those writes
        # leaves both baselines available for conservative recovery.
        self._on_state_pending = on_state_pending
        self._on_state_committed = on_state_committed
        self._transaction_lock = threading.Lock()
        self._synced_files: dict[str, tuple[float, int]] = {}  # remote_path -> (mtime, size)
        self._pushed_hashes: dict[str, str] = {}  # remote_path -> sha256 hex digest
        self._upload_only_host_paths: set[str] = set()
        self._recovery_hashes: dict[str, set[str]] = {}
        self._state_promotion_required = False
        self._has_successful_sync = False
        self._last_sync_time: float = 0.0  # monotonic; 0 ensures first sync runs
        self._sync_interval = sync_interval

    def sync(self, *, force: bool = False) -> bool:
        """Run a sync cycle: upload changed files, delete removed files.

        Rate-limited to once per ``sync_interval`` unless *force* is True
        or ``HERMES_FORCE_FILE_SYNC=1`` is set.

        Transactional: state only committed if ALL operations succeed.
        On failure, state rolls back so the next cycle retries everything.

        Returns ``True`` when the cycle succeeds (including when there is no
        work), and ``False`` when a transport operation fails and rolls back.
        """
        with self._transaction_lock:
            return self._sync_transaction(force=force)

    def reset_remote_state(self) -> None:
        """Forget committed transport state after the remote is replaced.

        The next sync uploads every current file even when none changed on the
        host. Content hashes from the old remote must not suppress sync-back
        comparisons against the replacement.
        """
        with self._transaction_lock:
            self._synced_files.clear()
            self._pushed_hashes.clear()
            self._recovery_hashes.clear()
            self._state_promotion_required = False
            self._has_successful_sync = False
            self._last_sync_time = 0.0

    def export_state(self) -> dict:
        """Return a JSON-serializable snapshot of committed transport state."""
        with self._transaction_lock:
            return self._export_state_locked()

    def _export_state_locked(self) -> dict:
        return self._serialize_state(
            self._synced_files,
            self._pushed_hashes,
            self._upload_only_host_paths,
        )

    @staticmethod
    def _serialize_state(
        synced_files: dict[str, tuple[float, int]],
        pushed_hashes: dict[str, str],
        upload_only_host_paths: set[str],
    ) -> dict:
        return {
            "synced_files": {
                remote: [key[0], key[1]] for remote, key in synced_files.items()
            },
            "pushed_hashes": dict(pushed_hashes),
            "upload_only_host_paths": sorted(upload_only_host_paths),
        }

    @staticmethod
    def _parse_state(
        state: object,
    ) -> tuple[dict[str, tuple[float, int]], dict[str, str], set[str]] | None:
        """Validate and normalize one serialized sync snapshot."""
        if not isinstance(state, dict):
            return None
        state_dict = cast(Mapping[str, object], state)
        synced_raw = state_dict.get("synced_files")
        hashes_raw = state_dict.get("pushed_hashes")
        if not isinstance(synced_raw, dict) or not isinstance(hashes_raw, dict):
            return None
        synced: dict[str, tuple[float, int]] = {}
        for remote, key in synced_raw.items():
            if not isinstance(remote, str):
                return None
            if (
                not isinstance(key, (list, tuple))
                or len(key) != 2
            ):
                return None
            mtime_raw, size_raw = key
            if not isinstance(mtime_raw, (int, float)) or not isinstance(
                size_raw, (int, float)
            ):
                return None
            synced[remote] = (float(mtime_raw), int(size_raw))
        hashes: dict[str, str] = {}
        for remote, digest in hashes_raw.items():
            if not isinstance(remote, str) or not isinstance(digest, str):
                return None
            hashes[remote] = digest
        upload_only_raw = state_dict.get("upload_only_host_paths")
        upload_only = (
            {path for path in upload_only_raw if isinstance(path, str)}
            if isinstance(upload_only_raw, list)
            else set()
        )
        return synced, hashes, upload_only

    def load_state(self, state: object, *, pending_state: object = None) -> bool:
        """Restore a committed-state snapshot produced by :meth:`export_state`.

        Returns ``True`` when *state* was valid and loaded. A loaded baseline
        marks the manager as having synced before, so a later
        :meth:`sync_back` can compare the remote against what the previous
        session last committed, and the next :meth:`sync` can detect files
        deleted on the host while the remote was paused.
        """
        pending = self._parse_state(pending_state)
        pending_mapping = (
            cast(Mapping[str, object], pending_state)
            if isinstance(pending_state, dict)
            else None
        )
        pending_operation = (
            pending_mapping.get(_PENDING_OPERATION_KEY)
            if pending_mapping is not None
            else None
        )
        parsed = self._parse_state(state)
        if parsed is None:
            if pending is not None:
                _pending_files, pending_hashes, pending_upload_only = pending
                with self._transaction_lock:
                    self._upload_only_host_paths |= pending_upload_only
                    if pending_operation == _PENDING_PUSH:
                        self._recovery_hashes = {
                            remote: {digest}
                            for remote, digest in pending_hashes.items()
                        }
            return False
        synced, hashes, upload_only = parsed

        with self._transaction_lock:
            self._synced_files = synced
            self._pushed_hashes = hashes
            self._upload_only_host_paths |= upload_only
            self._recovery_hashes = {}
            if pending is not None:
                _pending_files, pending_hashes, pending_upload_only = pending
                self._upload_only_host_paths |= pending_upload_only
                if pending_operation == _PENDING_PUSH:
                    for remote, digest in pending_hashes.items():
                        self._recovery_hashes.setdefault(remote, set()).add(digest)
            self._has_successful_sync = True
            self._last_sync_time = 0.0
        return True

    def _notify_state_committed(self) -> None:
        """Report the freshly committed state; must hold the transaction lock."""
        if self._on_state_committed is None:
            return
        self._on_state_committed(self._export_state_locked())

    def _notify_state_pending(self, state: dict) -> None:
        """Persist write-ahead state before mutating either side."""
        if self._on_state_pending is not None:
            self._on_state_pending(state)

    def _sync_transaction(self, *, force: bool = False) -> bool:
        """Execute one sync cycle while holding the per-manager lock."""
        if not force and not os.environ.get(_FORCE_SYNC_ENV):
            now = _monotonic()
            if (
                self._has_successful_sync
                and now - self._last_sync_time < self._sync_interval
            ):
                return True

        current_files = self._get_files_fn()
        prev_upload_only = set(self._upload_only_host_paths)
        self._upload_only_host_paths.update(_credential_host_paths())
        current_remote_paths = {remote for _, remote in current_files}

        # --- Uploads: new or changed files ---
        to_upload: list[tuple[str, str]] = []
        new_files = dict(self._synced_files)
        for host_path, remote_path in current_files:
            file_key = _file_mtime_key(host_path)
            if file_key is None:
                continue
            if self._synced_files.get(remote_path) == file_key:
                continue
            to_upload.append((host_path, remote_path))
            new_files[remote_path] = file_key

        # --- Deletes: synced paths no longer in current set ---
        to_delete = [p for p in self._synced_files if p not in current_remote_paths]

        if not to_upload and not to_delete:
            self._has_successful_sync = True
            self._last_sync_time = _monotonic()
            try:
                # Also clears a durable pending marker left by a successful
                # transport whose final metadata write failed.
                self._notify_state_committed()
                self._state_promotion_required = False
                return True
            except Exception as exc:
                self._last_sync_time = 0.0
                logger.warning("file_sync: committed-state persistence failed: %s", exc)
                return False

        # Snapshot for rollback (only when there's work to do)
        prev_files = dict(self._synced_files)
        prev_hashes = dict(self._pushed_hashes)
        candidate_hashes = dict(prev_hashes)

        try:
            for host_path, remote_path in to_upload:
                candidate_hashes[remote_path] = _sha256_file(host_path)
            for remote_path in to_delete:
                new_files.pop(remote_path, None)
                candidate_hashes.pop(remote_path, None)
            pending_state = self._serialize_state(
                new_files,
                candidate_hashes,
                self._upload_only_host_paths,
            )
            pending_state[_PENDING_OPERATION_KEY] = _PENDING_PUSH
            self._notify_state_pending(pending_state)
        except Exception as exc:
            self._upload_only_host_paths = prev_upload_only
            logger.warning(
                "file_sync: write-ahead state persistence failed; remote unchanged: %s",
                exc,
            )
            return False

        if to_upload:
            logger.debug("file_sync: uploading %d file(s)", len(to_upload))
        if to_delete:
            logger.debug("file_sync: deleting %d stale remote file(s)", len(to_delete))

        try:
            if to_upload and self._bulk_upload_fn is not None:
                self._bulk_upload_fn(to_upload)
                logger.debug("file_sync: bulk-uploaded %d file(s)", len(to_upload))
            else:
                for host_path, remote_path in to_upload:
                    self._upload_fn(host_path, remote_path)
                    logger.debug("file_sync: uploaded %s -> %s", host_path, remote_path)

            if to_delete:
                self._delete_fn(to_delete)
                logger.debug("file_sync: deleted %s", to_delete)

            self._synced_files = new_files
            self._pushed_hashes = candidate_hashes
            self._recovery_hashes.clear()
            self._state_promotion_required = True
            self._has_successful_sync = True
            self._last_sync_time = _monotonic()

        except Exception as exc:
            self._synced_files = prev_files
            self._pushed_hashes = prev_hashes
            self._upload_only_host_paths = prev_upload_only
            # Do NOT advance _last_sync_time here: a failed cycle rolls state
            # back so the next cycle can retry. Bumping the rate-limit clock on
            # failure would make the next non-forced sync() return early (the
            # guard above), suppressing that retry for up to _sync_interval and
            # leaving the remote with stale files — contradicting this method's
            # documented "next cycle retries everything" contract.
            logger.warning("file_sync: sync failed, rolled back state: %s", exc)
            return False

        try:
            self._notify_state_committed()
            self._state_promotion_required = False
            return True
        except Exception as exc:
            # Remote mutation completed and the write-ahead snapshot remains
            # durable. Keep the candidate in memory so the next cycle only
            # retries promotion; block command execution until that succeeds.
            self._last_sync_time = 0.0
            logger.warning("file_sync: committed-state persistence failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Sync-back: pull remote changes to host on teardown
    # ------------------------------------------------------------------

    def sync_back(
        self,
        hermes_home: Path | None = None,
        *,
        require_prior_sync: bool = True,
        restrict_to_roots: bool = False,
    ) -> bool:
        """Pull remote changes back to the host filesystem.

        Downloads the remote ``.hermes/`` directory as a tar archive,
        unpacks it, and applies only files that differ from what was
        originally pushed (based on SHA-256 content hashes).

        Protected against SIGINT (defers the signal until complete) and
        serialized across concurrent gateway sandboxes via file lock.

        ``require_prior_sync=False`` lets a backend pull from a remote this
        manager instance never pushed to — the resume-recovery path, where the
        remote was initialized by an earlier session whose committed baseline
        is unavailable. Every remote file then counts as remote-authored.

        ``restrict_to_roots=True`` maps remote files only through the declared
        ``sync_back_roots``. Backends use it together with
        ``require_prior_sync=False``: without a baseline nothing separates
        host-owned files (credentials, cache) from remote-authored ones, so
        only the explicit agent-state roots are safe to write to the host —
        general prefix inference could otherwise recreate a credential the
        host deleted.

        Returns ``True`` when the pull completed (or there was nothing to
        pull), ``False`` when it was skipped or every retry failed — the
        caller must then treat remote changes as not yet recovered.
        """
        with self._transaction_lock:
            return self._sync_back_transaction(
                hermes_home=hermes_home,
                require_prior_sync=require_prior_sync,
                restrict_to_roots=restrict_to_roots,
            )

    def _sync_back_transaction(
        self,
        hermes_home: Path | None = None,
        require_prior_sync: bool = True,
        restrict_to_roots: bool = False,
    ) -> bool:
        """Execute sync-back against a stable snapshot of manager state."""
        if self._bulk_download_fn is None:
            return True
        if sys.is_finalizing():
            logger.debug("sync_back: interpreter shutting down — skipping")
            return False

        # A successful no-work cycle still initializes an empty remote and may
        # later produce new agent state. Only skip when no sync cycle has ever
        # succeeded (for example, after a rolled-back initial upload).
        if require_prior_sync and not self._has_successful_sync:
            logger.debug("sync_back: no successful prior sync — skipping")
            return False

        lock_path = (hermes_home or get_hermes_home()) / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        last_exc: Exception | None = None
        for attempt in range(_SYNC_BACK_MAX_RETRIES):
            try:
                self._sync_back_once(lock_path, restrict_to_roots=restrict_to_roots)
                return True
            except _PermanentSyncBackError as exc:
                logger.warning("sync_back: rejected without retry: %s", exc)
                return False
            except Exception as exc:
                last_exc = exc
                if attempt < _SYNC_BACK_MAX_RETRIES - 1:
                    delay = _SYNC_BACK_BACKOFF[attempt]
                    logger.warning(
                        "sync_back: attempt %d failed (%s), retrying in %ds",
                        attempt + 1, exc, delay,
                    )
                    _sleep(delay)

        logger.warning("sync_back: all %d attempts failed: %s", _SYNC_BACK_MAX_RETRIES, last_exc)
        return False

    def _sync_back_once(self, lock_path: Path, *, restrict_to_roots: bool = False) -> None:
        """Single sync-back attempt with SIGINT protection and file lock."""
        # signal.signal() only works from the main thread. In gateway
        # contexts cleanup() may run from a worker thread — skip SIGINT
        # deferral there rather than crashing.
        on_main_thread = threading.current_thread() is threading.main_thread()

        deferred_sigint: list[object] = []
        original_handler = None
        if on_main_thread:
            original_handler = signal.getsignal(signal.SIGINT)

            def _defer_sigint(signum, frame):
                deferred_sigint.append((signum, frame))
                logger.debug("sync_back: SIGINT deferred until sync completes")

            signal.signal(signal.SIGINT, _defer_sigint)
        try:
            self._sync_back_locked(lock_path, restrict_to_roots=restrict_to_roots)
        finally:
            if on_main_thread and original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)
                if deferred_sigint:
                    # Re-deliver the deferred Ctrl+C to the just-restored
                    # handler. ``os.kill(os.getpid(), signal.SIGINT)`` is NOT a
                    # graceful signal on Windows: os.kill only treats
                    # CTRL_C_EVENT(0)/CTRL_BREAK_EVENT(1) as console events; any
                    # other value (SIGINT == 2) routes to TerminateProcess(sig),
                    # hard-killing the CLI (exit code 2) instead of raising
                    # KeyboardInterrupt — so a Ctrl+C during a remote-backend
                    # sync-back would kill the whole session on Windows.
                    # ``signal.raise_signal`` (3.8+) invokes the handler via C
                    # ``raise()`` on every platform.
                    signal.raise_signal(signal.SIGINT)

    def _sync_back_locked(self, lock_path: Path, *, restrict_to_roots: bool = False) -> None:
        """Sync-back under file lock (serializes concurrent gateways)."""
        if fcntl is None:
            # Windows: no flock — run without serialization
            self._sync_back_impl(restrict_to_roots=restrict_to_roots)
            return
        lock_fd = open(lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            self._sync_back_impl(restrict_to_roots=restrict_to_roots)
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except (OSError, IOError):
                pass
            lock_fd.close()

    def _sync_back_impl(self, *, restrict_to_roots: bool = False) -> None:
        """Download, diff, and apply remote changes to host."""
        if self._bulk_download_fn is None:
            raise RuntimeError("_sync_back_impl called without bulk_download_fn")

        # Cache file mapping once to avoid O(n*m) from repeated iteration
        try:
            file_mapping = list(self._get_files_fn())
        except Exception:
            file_mapping = []

        with tempfile.NamedTemporaryFile(suffix=".tar") as tf:
            self._bulk_download_fn(Path(tf.name))

            # Defensive size cap: a misbehaving sandbox could produce an
            # arbitrarily large tar. Refuse to extract if it exceeds the cap.
            try:
                tar_size = os.path.getsize(tf.name)
            except OSError:
                tar_size = 0
            if tar_size > _SYNC_BACK_MAX_BYTES:
                raise _PermanentSyncBackError(
                    f"remote tar is {tar_size} bytes (cap {_SYNC_BACK_MAX_BYTES})"
                )

            with tempfile.TemporaryDirectory(prefix="hermes-sync-back-") as staging:
                with tarfile.open(tf.name) as tar:
                    tar.extractall(staging, filter="data")

                mapped_remote_paths = {remote for _, remote in file_mapping}
                upload_only_host_paths = (
                    self._upload_only_host_paths | _credential_host_paths()
                )
                planned: list[tuple[str, str, str, str, bool]] = []
                for dirpath, _dirnames, filenames in os.walk(staging):
                    for fname in filenames:
                        staged_file = os.path.join(dirpath, fname)
                        rel = os.path.relpath(staged_file, staging)
                        remote_path = "/" + rel

                        pushed_hash = self._pushed_hashes.get(remote_path)
                        remote_hash = _sha256_file(staged_file)
                        known_hashes = set(self._recovery_hashes.get(remote_path, ()))
                        if pushed_hash is not None:
                            known_hashes.add(pushed_hash)
                        # A write-ahead snapshot may describe an upload that
                        # completed before the process died but whose final
                        # baseline promotion did not. Treat either the old or
                        # pending hash as host-authored, never as a remote edit.
                        if remote_hash in known_hashes:
                            continue

                        # Resolve host path from cached mapping
                        if restrict_to_roots:
                            host_path = self._map_via_sync_back_roots(
                                remote_path,
                                upload_only_host_paths=upload_only_host_paths,
                            )
                        else:
                            host_path = self._resolve_host_path(remote_path, file_mapping)
                            if host_path is None:
                                host_path = self._infer_host_path(
                                    remote_path,
                                    file_mapping,
                                    upload_only_host_paths=upload_only_host_paths,
                                )
                        if host_path is None:
                            logger.debug(
                                "sync_back: skipping %s (no host mapping)",
                                remote_path,
                            )
                            continue

                        if self._is_upload_only_host_path(host_path, upload_only_host_paths):
                            logger.debug(
                                "sync_back: skipping upload-only credential file %s",
                                remote_path,
                            )
                            continue

                        if os.path.exists(host_path) and pushed_hash is not None:
                            host_hash = _sha256_file(host_path)
                            if host_hash != pushed_hash:
                                logger.warning(
                                    "sync_back: conflict on %s — host modified "
                                    "since push, remote also changed. Applying "
                                    "remote version (last-write-wins).",
                                    remote_path,
                                )

                        update_inventory = (
                            remote_path in self._synced_files
                            or remote_path in mapped_remote_paths
                        )
                        planned.append(
                            (
                                staged_file,
                                host_path,
                                remote_path,
                                remote_hash,
                                update_inventory,
                            )
                        )

                if planned:
                    candidate_files = dict(self._synced_files)
                    candidate_hashes = dict(self._pushed_hashes)
                    for (
                        staged_file,
                        _host_path,
                        remote_path,
                        remote_hash,
                        update_inventory,
                    ) in planned:
                        candidate_hashes[remote_path] = remote_hash
                        if update_inventory:
                            # copy2 preserves the staged mtime and size on the
                            # host, so this is the post-copy inventory key.
                            host_key = _file_mtime_key(staged_file)
                            if host_key is not None:
                                candidate_files[remote_path] = host_key

                    pending_state = self._serialize_state(
                        candidate_files,
                        candidate_hashes,
                        upload_only_host_paths,
                    )
                    pending_state[_PENDING_OPERATION_KEY] = _PENDING_PULL
                    self._notify_state_pending(pending_state)

                    for (
                        staged_file,
                        host_path,
                        _remote_path,
                        _remote_hash,
                        _update_inventory,
                    ) in planned:
                        os.makedirs(os.path.dirname(host_path), exist_ok=True)
                        shutil.copy2(staged_file, host_path)

                    self._synced_files = candidate_files
                    self._pushed_hashes = candidate_hashes
                    self._upload_only_host_paths = upload_only_host_paths
                    self._recovery_hashes.clear()
                    self._state_promotion_required = True
                    logger.info("sync_back: applied %d changed file(s)", len(planned))
                else:
                    logger.debug("sync_back: no remote changes detected")

                if self._state_promotion_required:
                    self._notify_state_committed()
                    self._state_promotion_required = False

    def _resolve_host_path(self, remote_path: str,
                           file_mapping: list[tuple[str, str]] | None = None) -> str | None:
        """Find the host path for a known remote path from the file mapping."""
        mapping = file_mapping if file_mapping is not None else []
        for host, remote in mapping:
            if remote == remote_path:
                return host
        return None

    def _infer_host_path(self, remote_path: str,
                         file_mapping: list[tuple[str, str]] | None = None,
                         *,
                         upload_only_host_paths: set[str] | None = None) -> str | None:
        """Infer a host path for a new remote file by matching path prefixes.

        Uses the existing file mapping to find a remote->host directory pair,
        then applies the same prefix substitution to the new file. Backends
        may also declare narrow sync-back roots for initially empty state
        directories.
        For example, if the mapping has ``/root/.hermes/skills/a.md`` →
        ``~/.hermes/skills/a.md``, a new remote file at
        ``/root/.hermes/skills/b.md`` maps to ``~/.hermes/skills/b.md``.
        """
        mapping = file_mapping if file_mapping is not None else []
        upload_only_host_paths = upload_only_host_paths or set()
        for host, remote in mapping:
            if self._is_upload_only_host_path(host, upload_only_host_paths):
                continue
            remote_dir = posixpath.dirname(remote)
            if remote_path.startswith(remote_dir + "/"):
                host_dir = str(Path(host).parent)
                suffix = remote_path[len(remote_dir):]
                return host_dir + suffix

        # An empty local skill/memory directory produces no file mapping, and
        # a newly-created nested directory shares no file-level prefix with an
        # existing sibling. Backends can declare the specific state roots that
        # are safe to recreate on the host for those cases.
        return self._map_via_sync_back_roots(
            remote_path, upload_only_host_paths=upload_only_host_paths
        )

    def _map_via_sync_back_roots(self, remote_path: str,
                                 *,
                                 upload_only_host_paths: set[str] | None = None) -> str | None:
        """Map a remote path to the host through the declared sync-back roots.

        The roots are the only mapping used when no committed baseline exists
        (recovery of an unknown remote): they name exactly the agent-authored
        state directories, so host-owned files elsewhere can never be
        recreated through prefix inference.
        """
        upload_only_host_paths = upload_only_host_paths or set()
        normalized_remote = posixpath.normpath(remote_path)
        for host_root, remote_root in self._sync_back_roots:
            prefix = remote_root.rstrip("/") + "/"
            if not normalized_remote.startswith(prefix):
                continue

            relative = normalized_remote[len(prefix):]
            parts = PurePosixPath(relative).parts
            if not parts or any(part in {"", ".", ".."} for part in parts):
                continue

            candidate = host_root.joinpath(*parts).resolve()
            if candidate != host_root and host_root not in candidate.parents:
                logger.warning(
                    "sync_back: refusing %s because it escapes host root %s",
                    remote_path,
                    host_root,
                )
                continue
            candidate_str = str(candidate)
            if self._is_upload_only_host_path(candidate_str, upload_only_host_paths):
                continue
            return candidate_str
        return None

    @staticmethod
    def _is_upload_only_host_path(host_path: str, upload_only_host_paths: set[str]) -> bool:
        try:
            resolved = str(Path(host_path).expanduser().resolve())
        except OSError:
            resolved = str(Path(host_path).expanduser())
        return resolved in upload_only_host_paths
