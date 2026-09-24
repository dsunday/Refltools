"""
jobs.py
-------
Background job records and launching.

Each job is jobs/<id>.json:
    {id, kind, models, energies, params, gpu, status, pid, log, created,
     started, finished, error, outputs}
status: queued → running → done | failed | cancelled   (crashed if the
process disappeared while 'running').

The job runs in a fresh process (python -m rsoxr_harness _worker ...) so the
GPU choice (CUDA_VISIBLE_DEVICES) applies cleanly and the caller is free.
"""

import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime

TERMINAL = ("done", "failed", "cancelled", "crashed")


def _now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# h5 lock – serialises writers; readers take a shared lock
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def h5_lock(h5_path, shared=False):
    lock_path = h5_path + ".lock"
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# job records
# ---------------------------------------------------------------------------

def _job_path(project, job_id):
    return os.path.join(project.dir("jobs"), f"{job_id}.json")


def read_job(project, job_id):
    with open(_job_path(project, job_id)) as fh:
        return json.load(fh)


def write_job(project, job):
    d = project.dir("jobs")
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".job.", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(job, fh, indent=2)
    os.replace(tmp, _job_path(project, job["id"]))


def update_job(project, job_id, **kw):
    job = read_job(project, job_id)
    job.update(kw)
    write_job(project, job)
    return job


def _alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie child of this process still "exists"; reap if possible
    try:
        wpid, _ = os.waitpid(pid, os.WNOHANG)
        return wpid == 0
    except ChildProcessError:
        return True


def list_jobs(project, refresh=True):
    jobs = []
    for fn in sorted(os.listdir(project.dir("jobs"))):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        job = read_job(project, fn[:-5])
        if refresh and job["status"] in ("queued", "running") and job.get("pid") \
                and not _alive(job["pid"]):
            # re-read: the worker may have finished between the two reads
            job = read_job(project, job["id"])
            if job["status"] in ("queued", "running"):
                job = update_job(project, job["id"], status="crashed",
                                 finished=_now(),
                                 error="process exited without updating the job "
                                       "record (see log)")
        jobs.append(job)
    return jobs


def new_job(project, kind, models, energies, params, gpu):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "-".join(models) if len(models) <= 3 else f"{models[0]}+{len(models) - 1}"
    job_id = f"{stamp}_{kind}_{tag}"
    n = 1
    while os.path.exists(_job_path(project, job_id)):
        n += 1
        job_id = f"{stamp}_{kind}_{tag}_{n}"
    job = {
        "id": job_id, "kind": kind, "models": list(models),
        "energies": [float(e) for e in energies], "params": params,
        "gpu": gpu, "status": "queued", "pid": None,
        "log": os.path.join(project.dir("logs"), f"{job_id}.log"),
        "created": _now(), "started": None, "finished": None,
        "error": None, "outputs": {},
    }
    write_job(project, job)
    return job


# ---------------------------------------------------------------------------
# launching
# ---------------------------------------------------------------------------

def busy_gpus(project):
    return {j["gpu"] for j in list_jobs(project)
            if j["status"] in ("queued", "running")}


def launch(project, job, detach=True, echo=True):
    """
    Start the worker for a queued job.  gpu = int index, or "cpu".
    detach=False waits for completion, streaming the log to stdout.
    """
    env = dict(os.environ)
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["MPLBACKEND"] = "Agg"
    if job["gpu"] == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("JAX_PLATFORMS", None)
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    refltools = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = refltools + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [sys.executable, "-m", "rsoxr_harness", "_worker",
           "--project", project.path, "--job", job["id"]]
    log = open(job["log"], "a")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=refltools, env=env, start_new_session=True)
    log.close()
    update_job(project, job["id"], pid=proc.pid)
    if detach:
        return proc

    # foreground: follow the log until the process ends
    with open(job["log"]) as fh:
        while True:
            chunk = fh.read()
            if chunk and echo:
                sys.stdout.write(chunk)
                sys.stdout.flush()
            if proc.poll() is not None:
                rest = fh.read()
                if rest and echo:
                    sys.stdout.write(rest)
                break
            time.sleep(1.0)
    return proc


def cancel(project, job_id):
    job = read_job(project, job_id)
    if job["status"] not in ("queued", "running"):
        return job
    if job.get("pid") and _alive(job["pid"]):
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    return update_job(project, job_id, status="cancelled", finished=_now())


def tail(path, n=20):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 64 * 1024))
        lines = fh.read().decode(errors="replace").splitlines()
    # collapse carriage-return progress bars
    lines = [l.split("\r")[-1] for l in lines]
    return "\n".join(lines[-n:])


def _elapsed(job):
    if not job.get("started"):
        return ""
    t0 = datetime.fromisoformat(job["started"])
    t1 = datetime.fromisoformat(job["finished"]) if job.get("finished") else datetime.now()
    s = int((t1 - t0).total_seconds())
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def status_table(project, job_id=None, n_tail=0):
    jobs = list_jobs(project)
    if job_id:
        jobs = [j for j in jobs if j["id"] == job_id]
        if not jobs:
            return f"no job '{job_id}'"
    if not jobs:
        return "(no jobs)"
    rows = [("job", "status", "gpu", "energies", "elapsed")]
    for j in jobs:
        rows.append((j["id"], j["status"], str(j["gpu"]), str(len(j["energies"])),
                     _elapsed(j)))
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = ["  ".join(c.ljust(w[i]) for i, c in enumerate(r)) for r in rows]
    for j in jobs:
        if j.get("error") and (job_id or j["status"] in ("failed", "crashed")):
            out.append(f"\n{j['id']} error: {j['error'].strip().splitlines()[-1]}")
        if n_tail:
            out.append(f"\n--- {j['id']} log (last {n_tail} lines) ---")
            out.append(tail(j["log"], n_tail))
    return "\n".join(out)
