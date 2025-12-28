import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional
import os

from fastapi import FastAPI, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from src.models.tft.infer import infer_tft_from_dict
from src.inference.tcn_infer import infer_single, load_infer_config_from_dict


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
        self.max_logs = 2000

    def start(self, job: str, script: str, extra_args: Optional[List[str]] = None) -> None:
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise RuntimeError("已有任务在运行")
            self.logs.append(f"--- 开始任务: {job} ---")
            self.status = "running"
            self.current_job = job
            self.last_job = job
            cmd = [sys.executable, str(SCRIPTS / script)]
            if extra_args:
                cmd.extend(extra_args)
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            threading.Thread(target=self._collect_logs, daemon=True).start()

    def _collect_logs(self) -> None:
        assert self.process is not None
        for line in self.process.stdout:
            with self.lock:
                self.logs.append(line.rstrip())
                if len(self.logs) > self.max_logs:
                    self.logs = self.logs[-self.max_logs :]
        code = self.process.wait()
        with self.lock:
            if code == 0:
                self.status = "completed"
            else:
                self.status = "failed"
            self.current_job = None
            self.logs.append(f"--- 任务结束: {self.last_job} ({self.status}) ---")

    def stop(self) -> None:
        with self.lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()
                self.status = "stopped"
                self.current_job = None
            elif self.process is None or self.process.poll() is not None:
                self.status = "stopped"
                self.current_job = None
            self.logs.append("--- 已停止当前任务 ---")

    def add_log(self, message: str) -> None:
        with self.lock:
            self.logs.append(message)
            if len(self.logs) > self.max_logs:
                self.logs = self.logs[-self.max_logs :]

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
def start_train(payload: dict = Body(default={})):
    model = str(payload.get("model", "tft")).lower()
    if model not in {"tft", "tcn"}:
        raise HTTPException(status_code=400, detail="Unsupported model")
    try:
        script = "train_tft.py" if model == "tft" else "train_tcn.py"
        jobs.start("train", script)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.post("/api/evaluate")
def start_eval(payload: dict = Body(default={})):
    model = str(payload.get("model", "tft")).lower()
    if model not in {"tft", "tcn"}:
        raise HTTPException(status_code=400, detail="Unsupported model")
    try:
        script = "evaluate_tft.py" if model == "tft" else "evaluate_tcn.py"
        jobs.start("evaluate", script)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"status": "started"}


@app.post("/api/infer")
def start_infer(payload: dict = Body(default={})):
    model = str(payload.get("model", "tft")).lower()
    if model not in {"tft", "tcn"}:
        raise HTTPException(status_code=400, detail="Unsupported model")
    base = ROOT / "configs" / "tft" / "infer.json" if model == "tft" else ROOT / "configs" / "tcn_infer.json"
    if base.exists():
        config = json.loads(base.read_text(encoding="utf-8"))
    else:
        config = {}
    if payload:
        config.update(payload)
    jobs.add_log("--- 开始任务: infer ---")
    try:
        if model == "tft":
            result = infer_tft_from_dict(config)
        else:
            result = infer_single(load_infer_config_from_dict(config))
        jobs.add_log(f"推理完成: {result.get('series_id')} step={result.get('horizon_step')}")
    except ValueError as exc:
        jobs.add_log(f"推理失败: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        jobs.add_log(f"推理失败: {exc}")
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return result


@app.post("/api/prepare")
def start_prepare(payload: dict = Body(default={})):
    rows = int(payload.get("sample_rows", 100000))
    try:
        jobs.start("prepare", "prepare_dataset.py", ["--sample-rows", str(rows)])
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
