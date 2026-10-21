"""
_portable.py
------------
The few OS-specific operations the job runner needs, for POSIX and Windows:
file locks, "is this pid alive", detached launch, and killing a job's process tree.
POSIX behaviour is unchanged from the original fcntl/killpg code.
"""

import contextlib
import os
import subprocess
import time

WINDOWS = os.name == "nt"

if WINDOWS:
    import msvcrt
else:
    import fcntl


@contextlib.contextmanager
def file_lock(fh, shared=False):
    """
    Lock the open file `fh` for the duration of the block.  POSIX: flock (shared or
    exclusive).  Windows: msvcrt has no shared locks, so readers lock exclusively too
    (they just serialise); the first byte is locked, retrying until it is free.
    """
    if not WINDOWS:
        fcntl.flock(fh, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
        return
    fh.seek(0)
    while True:
        try:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            break
        except OSError:
            time.sleep(0.05)
    try:
        yield
    finally:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)


def pid_alive(pid):
    """True if process `pid` exists.  Never signals it (os.kill(pid, 0) on Windows
    would send CTRL_C_EVENT, not a probe)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if WINDOWS:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = wintypes.HANDLE
        PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return ctypes.get_last_error() == 5      # ERROR_ACCESS_DENIED: exists
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def detached_popen_kwargs():
    """Popen kwargs so a job survives Ctrl+C / closing the launching terminal."""
    if WINDOWS:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def terminate_tree(pid):
    """Stop a job's worker and its children (stoich/escan process pools)."""
    if WINDOWS:
        # no SIGTERM delivery to a detached process: force-kill the whole tree
        subprocess.run(["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    import signal
    try:
        os.killpg(int(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
