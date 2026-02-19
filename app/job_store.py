"""Simple in-process job store for background analytics tasks."""
import uuid
from dataclasses import dataclass, field
from enum import Enum
from time import monotonic
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


@dataclass
class Job:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=monotonic)


_STORE: dict[str, Job] = {}
_TTL = 300  # seconds to keep completed jobs


def create_job() -> Job:
    job = Job(job_id=str(uuid.uuid4()))
    _STORE[job.job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _STORE.get(job_id)


def set_running(job_id: str) -> None:
    if job_id in _STORE:
        _STORE[job_id].status = JobStatus.RUNNING


def set_done(job_id: str, result: Any) -> None:
    if job_id in _STORE:
        _STORE[job_id].status = JobStatus.DONE
        _STORE[job_id].result = result


def set_error(job_id: str, error: str) -> None:
    if job_id in _STORE:
        _STORE[job_id].status = JobStatus.ERROR
        _STORE[job_id].error = error


def purge_old() -> None:
    """Remove completed jobs older than TTL."""
    now = monotonic()
    to_delete = [
        jid for jid, job in _STORE.items()
        if job.status in (JobStatus.DONE, JobStatus.ERROR)
        and (now - job.created_at) > _TTL
    ]
    for jid in to_delete:
        del _STORE[jid]
