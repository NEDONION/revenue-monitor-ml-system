import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


class JobManager:
    # 管理训练/评估/推理任务的状态与日志。
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: Optional[subprocess.Popen] = None
        self.logs: List[str] = []
        self.status = "idle"
        self.current_job: Optional[str] = None
        self.last_job: Optional[str] = None

    def start(self, job: str, script: str, extra_args: Optional[List[str]] = None) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("已有任务在运行")
            self.logs.clear()
            self.status = "running"
            self.current_job = job
            self.last_job = job
            cmd = [sys.executable, str(SCRIPTS / script)]
            if extra_args:
                cmd.extend(extra_args)
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            threading.Thread(target=self._collect_logs, daemon=True).start()

    def _collect_logs(self) -> None:
        assert self.process is not None
        for line in self.process.stdout:
            with self.lock:
                self.logs.append(line.rstrip())
        code = self.process.wait()
        with self.lock:
            if code == 0:
                self.status = "completed"
            else:
                self.status = "failed"
            self.current_job = None

    def stop(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.status = "stopped"
                self.current_job = None
            elif self.process is None or self.process.poll() is not None:
                self.status = "stopped"
                self.current_job = None

    def tail(self, start: int) -> dict:
        with self.lock:
            lines = self.logs[start:]
            return {
                "lines": lines,
                "next": len(self.logs),
                "status": self.status,
                "job": self.current_job or self.last_job,
            }


app = FastAPI()
jobs = JobManager()
app.mount("/reports", StaticFiles(directory=ROOT / "reports"), name="reports")
app.mount("/models", StaticFiles(directory=ROOT / "models"), name="models")


@app.post("/api/train")
def start_train():
    try:
        jobs.start("train", "train_tcn.py")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.post("/api/evaluate")
def start_eval():
    try:
        jobs.start("evaluate", "evaluate_tcn.py")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.post("/api/infer")
def start_infer(payload: dict = Body(default={})):
    try:
        config_path = ROOT / "reports" / "last_infer.json"
        base = ROOT / "configs" / "tcn_infer.json"
        if base.exists():
            config = json.loads(base.read_text(encoding="utf-8"))
        else:
            config = {}
        if payload:
            config.update(payload)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        jobs.start("infer", "run_inference.py", ["--config", str(config_path)])
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.post("/api/stop")
def stop_job():
    jobs.stop()
    return {"status": "stopped"}


@app.get("/api/logs")
def get_logs(start: int = 0):
    return jobs.tail(start)


@app.get("/api/status")
def get_status():
    snapshot = jobs.tail(0)
    return {"status": snapshot["status"], "job": snapshot["job"]}
