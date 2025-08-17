from __future__ import annotations

import json
import threading
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import logging

from app.config import settings
from app.services.ocr import ocr_on_cropped_image, draw_overlay_on_image
from app.services.llm import extract_fields_from_text
from app.services.db import insert_llm_run


logger = logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    stored_name: str
    original_name: str
    status: str = "queued"  # queued | processing | done | failed
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    # When true, recompute artifacts even if they already exist
    force: bool = False


class JobManager:
    def __init__(self) -> None:
        self._q: queue.Queue[Job] = queue.Queue()
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._worker, name="receipt-worker", daemon=True)
        self._thread.start()

    def enqueue(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.id] = job
        self._q.put(job)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> Dict[str, Job]:
        with self._lock:
            return dict(self._jobs)

    def any_busy(self) -> bool:
        with self._lock:
            for j in self._jobs.values():
                if j.status in {"queued", "processing"}:
                    return True
        return False

    def clear_jobs(self) -> None:
        """Clear tracked job records (does not cancel running tasks)."""
        with self._lock:
            self._jobs.clear()

    def _worker(self) -> None:
        while True:
            job = self._q.get()
            if not job:
                continue
            try:
                job.status = "processing"
                job.started_at = time.time()
                self._process(job)
                job.status = "done"
                job.finished_at = time.time()
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                job.finished_at = time.time()
                logger.exception("Job failed: %s", job.id)
            finally:
                self._q.task_done()

    def _process(self, job: Job) -> None:
        stored_path = settings.UPLOADS_DIR / job.stored_name
        if not stored_path.exists():
            raise FileNotFoundError(f"uploaded file missing: {stored_path}")

        # If already processed, skip
        stem = Path(job.stored_name).stem
        fields_path = settings.PROCESSED_DIR / f"fields_{stem}.json"
        overlay_path = settings.PROCESSED_DIR / f"overlay_{stem}.jpg"
        ocr_json_path = settings.PROCESSED_DIR / f"ocr_{stem}.json"
        if (fields_path.exists() and overlay_path.exists() and ocr_json_path.exists()) and not job.force:
            logger.info("Artifacts already exist for %s; skipping processing (force=%s)", job.stored_name, job.force)
            return

        # Run OCR
        import importlib
        cv2 = importlib.import_module("cv2")
        img_bgr = cv2.imread(str(stored_path))
        if img_bgr is None:
            raise RuntimeError("Failed to read image for processing")
        debug_base = Path(job.stored_name).stem
        detailed = ocr_on_cropped_image(img_bgr, debug_basename=debug_base)
        raw_text = detailed.get("text", "")

        # LLM extraction
        fields = extract_fields_from_text(raw_text, detailed.get("lines", []))
        used_model = settings.OLLAMA_MODEL
        try:
            if fields.get("metrics"):
                insert_llm_run(job.stored_name, used_model, fields["metrics"])  # type: ignore[arg-type]
        except Exception:
            logger.debug("llm_run insert failed for %s", job.stored_name)

        # Save overlay
        try:
            proc_image = detailed.get("proc_image")
            if proc_image is None:
                import importlib
                cv2 = importlib.import_module("cv2")
                proc_image = cv2.imread(str(stored_path))
            # Prefer the original color image as the base if same size, to avoid inverted-looking overlays
            import importlib
            cv2 = importlib.import_module("cv2")
            orig_img = cv2.imread(str(stored_path))

            def _wh(img):
                if img is None:
                    return (-1, -1)
                if len(img.shape) == 2:
                    h, w = img.shape
                else:
                    h, w = img.shape[:2]
                return (w, h)

            base_img = proc_image
            if orig_img is not None and _wh(orig_img) == _wh(proc_image):
                base_img = orig_img
            overlay = draw_overlay_on_image(base_img, detailed["words"], detailed["line_ids"], {})
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlay_path), overlay)
        except Exception as e:
            logger.warning("Failed to write overlay for %s: %s", job.stored_name, e)

        # Save OCR JSON
        try:
            serial = {
                "text": detailed.get("text", ""),
                "lines": detailed.get("lines", []),
                "words": detailed.get("words", []),
                "line_ids": detailed.get("line_ids", []),
                "size": list(detailed.get("size", []))
                if isinstance(detailed.get("size"), (list, tuple))
                else detailed.get("size"),
            }
            ocr_json_path.write_text(json.dumps(serial, indent=2))
        except Exception as e:
            logger.warning("Failed to write OCR JSON for %s: %s", job.stored_name, e)

        # Save fields JSON
        try:
            fields_doc = {
                "fields": fields,
                "used_model": used_model,
                "stored_name": job.stored_name,
                "original_name": job.original_name,
                "created_at": time.time(),
            }
            fields_path.write_text(json.dumps(fields_doc, indent=2))
        except Exception as e:
            logger.warning("Failed to write fields JSON for %s: %s", job.stored_name, e)


# Singleton Job Manager
job_manager = JobManager()
