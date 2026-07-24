"""
FastAPI server exposing the georelating pipeline as a demo API with a map UI.

Run from the repository root:
    uvicorn demo.server:app --host 0.0.0.0 --port 8123
"""
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from demo.service import GeorelatingService, MAX_TEXT_LENGTH, STAGE_ORDER, STAGE_LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("georelating.demo")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Persistent, append-only record of every invocation (one JSON object per line).
INVOCATION_LOG = os.path.join(PROJECT_ROOT, "output", "demo", "invocations.jsonl")
_invocation_log_lock = threading.Lock()


def _log_invocation(record: dict):
    """Append one invocation record to the persistent JSONL log."""
    try:
        os.makedirs(os.path.dirname(INVOCATION_LOG), exist_ok=True)
        with _invocation_log_lock:
            with open(INVOCATION_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception("Could not write invocation log")


app = FastAPI(title="Georelating Demo API",
              description="LLM Agents for Georelating — locate disaster-affected areas "
                          "from natural language reports as DGGS cells.")

service = GeorelatingService()
executor = ThreadPoolExecutor(max_workers=int(os.getenv("DEMO_MAX_WORKERS", "2")))
jobs = {}
jobs_lock = threading.Lock()


class GeorelateRequest(BaseModel):
    article_id: Optional[int] = Field(default=None, description="GANDR sample article id")
    text: Optional[str] = Field(default=None, max_length=MAX_TEXT_LENGTH,
                                description="User-provided disaster report text")
    use_structured_output: bool = Field(default=False,
                                        description="Constrain actor LLM calls to valid JSON "
                                                    "via the API's structured output mode")
    geonames_username: Optional[str] = Field(default=None, max_length=100,
                                             description="Optional GeoNames account (public "
                                                         "identifier) to use instead of the "
                                                         "server default, to draw on your own "
                                                         "GeoNames quota")


def _job_source(request: GeorelateRequest) -> str:
    return f"article_id={request.article_id}" if request.article_id is not None \
        else f"text[{len(request.text or '')} chars]"


def _run_job(job_id: str, request: GeorelateRequest):
    t0 = time.time()
    with jobs_lock:
        jobs[job_id].update({"status": "running", "started_at": t0})
    logger.info("Job %s started | %s | structured=%s", job_id, _job_source(request),
                request.use_structured_output)

    # Track stage transitions to record per-stage durations.
    timeline = []          # list of {stage, started_s, duration_s}
    stage_state = {"current": None, "t": t0}

    def on_progress(progress: dict):
        now = time.time()
        stage = progress.get("current")
        with jobs_lock:
            jobs[job_id]["progress"] = progress
        if stage != stage_state["current"]:
            if stage_state["current"] is not None:
                elapsed = now - stage_state["t"]
                timeline[-1]["duration_s"] = round(elapsed, 1)
                logger.info("Job %s | stage '%s' done in %.1fs (t+%.1fs)",
                            job_id, stage_state["current"], elapsed, now - t0)
            timeline.append({"stage": stage, "started_s": round(now - t0, 1), "duration_s": None})
            stage_state.update(current=stage, t=now)
            if progress.get("reflection_active"):
                logger.info("Job %s | reflection round in stage '%s' (t+%.1fs)", job_id, stage, now - t0)
            with jobs_lock:
                jobs[job_id]["timeline"] = [dict(s) for s in timeline]

    try:
        result = service.georelate(text=request.text,
                                   article_id=request.article_id,
                                   use_structured_output=request.use_structured_output,
                                   geonames_username=request.geonames_username,
                                   progress_callback=on_progress)
        t_end = time.time()
        if timeline:
            timeline[-1]["duration_s"] = round(t_end - stage_state["t"], 1)
        duration = round(t_end - t0, 1)
        georelated_ok = bool(result.get("georelated"))
        with jobs_lock:
            jobs[job_id].update({"status": "completed", "result": result, "finished_at": t_end,
                                 "duration_s": duration, "timeline": [dict(s) for s in timeline]})
        logger.info("Job %s completed in %.1fs | georelated=%s | %d toponym(s) geocoded",
                    job_id, duration, georelated_ok, len(result.get("geocoded_toponyms", [])))
        _log_invocation({
            "job_id": job_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": _job_source(request), "use_structured_output": request.use_structured_output,
            "actor_model": service.actor_model, "critic_model": service.critic_model,
            "status": "completed", "duration_s": duration, "georelated": georelated_ok,
            "n_geocoded": len(result.get("geocoded_toponyms", [])),
            "n_errors": len(result.get("errors") or []), "timeline": timeline,
        })
    except Exception as e:
        t_end = time.time()
        duration = round(t_end - t0, 1)
        with jobs_lock:
            jobs[job_id].update({"status": "failed", "error": str(e), "finished_at": t_end,
                                 "duration_s": duration, "timeline": [dict(s) for s in timeline]})
        logger.exception("Job %s failed after %.1fs", job_id, duration)
        _log_invocation({
            "job_id": job_id, "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": _job_source(request), "use_structured_output": request.use_structured_output,
            "actor_model": service.actor_model, "critic_model": service.critic_model,
            "status": "failed", "duration_s": duration, "error": str(e), "timeline": timeline,
        })


@app.on_event("startup")
def warm_up():
    # Load the article corpus immediately; defer LLM/NER init to a background
    # thread so the server binds its port quickly.
    threading.Thread(target=service.warm_up, daemon=True).start()


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {
        "stages": [{"key": k, "label": STAGE_LABELS[k]} for k in STAGE_ORDER],
        "actor_model": service.actor_model,
        "critic_model": service.critic_model,
    }


@app.get("/api/articles")
def list_articles():
    return service.list_articles()


@app.get("/api/articles/{article_id}")
def get_article(article_id: int):
    article = service.get_article(article_id)
    if article is None:
        raise HTTPException(status_code=404, detail="Unknown article id")
    return article


@app.post("/api/georelate")
def georelate(request: GeorelateRequest):
    if request.article_id is None and not (request.text and request.text.strip()):
        raise HTTPException(status_code=422, detail="Provide either article_id or text.")
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "result": None, "error": None, "progress": None,
                        "created_at": time.time(), "timeline": []}
    logger.info("Job %s queued | %s", job_id, _job_source(request))
    executor.submit(_run_job, job_id, request)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id")
        return dict(job)


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
