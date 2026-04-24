import asyncio
import json
import logging
import multiprocessing as mp
import os
import threading
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any

from dotenv import load_dotenv

from backend.lib.asset_manager.base import AssetManager
from backend.lib.asset_manager.factory import AssetManagerFactory
from backend.lib.job_manager.base import JobManager, JobQueue
from backend.lib.photobook.job_processor import JobProcessor
from backend.lib.redis.client import RedisClient
from backend.logging_utils import configure_logging_env
from backend.path_manager import PathManager

configure_logging_env()

# Load environment-specific file
env_file = ".env.prod" if os.getenv("ENV") == "production" else ".env.dev"
loaded = load_dotenv(dotenv_path=PathManager().get_repo_root() / env_file)
assert loaded, "Env not loaded"

MAX_JOB_TIMEOUT_SECS = 600  # 10 mins
SEND_HEARTBEAT_EVERY_SECS = 1
POLL_SHUTDOWN_EVERY_SECS = 1


class WorkerProcess(mp.Process):
    def __init__(self, heartbeat_connection: Connection, name: str = "worker"):
        super().__init__()
        self.name = name
        self.heartbeat_connection = heartbeat_connection

    def run(self) -> None:
        try:
            redis = RedisClient()
            job_manager = JobManager(redis, JobQueue.MAIN_TASK_QUEUE)
            asset_manager = AssetManagerFactory.create()

            def send_heartbeat() -> None:
                while True:
                    try:
                        self.heartbeat_connection.send("ping")
                        time.sleep(SEND_HEARTBEAT_EVERY_SECS)
                    except Exception:
                        break  # parent closed pipe

            threading.Thread(target=send_heartbeat, daemon=True).start()

            asyncio.run(self._main_loop(job_manager, asset_manager))
        except Exception as e:
            logging.exception(f"[{self.name}] Worker crashed: {e}")

    async def _main_loop(
        self, job_manager: JobManager, asset_manager: AssetManager
    ) -> None:
        logging.info(f"[{self.name}] Started worker process (PID={self.pid})")
        while True:
            # 1. Check for shutdown message
            if self.heartbeat_connection.poll(timeout=POLL_SHUTDOWN_EVERY_SECS):
                try:
                    msg = self.heartbeat_connection.recv()
                    if msg == "shutdown":
                        logging.info(f"[{self.name}] Received shutdown signal")
                        break
                except EOFError:
                    logging.warning(f"[{self.name}] Heartbeat pipe closed")
                    break

            try:
                task = await job_manager.dequeue(timeout=5)
                if not task:
                    continue  # No job this cycle

                _, payload = task

                try:
                    job_data = json.loads(payload)
                except json.JSONDecodeError:
                    logging.warning(f"[{self.name}] Invalid payload: {payload}")
                    raise
                if "job_id" not in job_data:
                    raise ValueError("Missing job_id in task payload")

                job_id = job_data["job_id"]

                try:
                    await asyncio.wait_for(
                        self._handle_task(job_id, job_data, job_manager, asset_manager),
                        timeout=MAX_JOB_TIMEOUT_SECS,
                    )
                except asyncio.TimeoutError:
                    logging.warning(
                        f"[{self.name}] Job timed out after {MAX_JOB_TIMEOUT_SECS}s, "
                        f"job_id: {job_id} payload: {payload}"
                    )
                    await job_manager.update_status(job_id, "error", error="Timeout")
            except Exception as e:
                logging.exception(f"[{self.name}] Unexpected worker error: {e}")

        logging.info(f"[{self.name}] Exiting cleanly")

    async def _handle_task(
        self,
        job_id: str,
        job_data: dict[str, Any],
        job_manager: JobManager,
        asset_manager: AssetManager,
    ) -> None:
        try:
            image_keys = job_data.get("image_keys", [])
            logging.info(
                f"[{self.name}] Processing job {job_id} with {len(image_keys)} images"
            )
            await job_manager.update_status(job_id, "processing")
            processor = JobProcessor(job_id, job_data, asset_manager)
            try:
                result = await processor.process()
            except Exception as e:
                logging.exception(
                    f"[{self.name}] Processor failed for job {job_id}: {e}"
                )
                await job_manager.update_status(job_id, "error", error=str(e))
                return

            await job_manager.update_status(job_id, "done", result=json.dumps(result))
            logging.info(f"[{self.name}] Job {job_id} completed with result: {result}")

        except Exception as e:
            traceback.print_exc()
            logging.warning(f"[{self.name}] Failed job {job_id}: {e}")
            await job_manager.update_status(job_id, "error", error=str(e))
