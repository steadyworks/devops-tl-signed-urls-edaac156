# job_manager.py
import json
import os
import time
from enum import Enum
from typing import Any, Literal, Optional

from backend.lib.redis.client import RedisClient


class JobQueue(Enum):
    MAIN_TASK_QUEUE = "main_task_queue"


class JobManager:
    @staticmethod
    def __get_queue_name(queue: JobQueue) -> str:
        prefix = "PROD_" if os.getenv("ENV") == "production" else "DEV_"
        return prefix + str(queue)

    def __init__(self, redis_client: RedisClient, queue: JobQueue) -> None:
        self.redis = redis_client
        self.queue_name = JobManager.__get_queue_name(queue)

    async def enqueue(self, job_id: str, image_keys: list[str]) -> str:
        # Enqueue task
        await self.redis.client.rpush(
            self.queue_name,
            json.dumps(
                {
                    "job_id": job_id,
                    "image_keys": image_keys,
                    "instruction": "I'm celeberating a memory with my girlfriend. Please help me create a photobook.",  ### FIXME
                }
            ),
        )

        # Initialize status
        # TODO: migrate to PostgresSQL
        await self.redis.client.hset(
            f"{job_id}",
            mapping={
                "status": "queued",
                "created_at": str(int(time.time())),
            },
        )
        return job_id

    async def dequeue(self, timeout: Optional[int] = 0) -> Optional[tuple[str, str]]:
        return await self.redis.client.blpop(self.queue_name, timeout=timeout)

    async def update_status(
        self,
        job_id: str,
        status: Literal["processing", "done", "error"],
        error: Optional[str] = None,
        result: Optional[str] = None,
    ) -> None:
        update_fields = {
            "status": status,
            "updated_at": str(int(time.time())),
        }
        if error:
            update_fields["error"] = error
        if result:
            update_fields["result"] = result

        # TODO: migrate to PostgresSQL
        await self.redis.client.hset(f"job:{job_id}", mapping=update_fields)  # type: ignore[arg-type]

    async def get_status(self, job_id: str) -> dict[str, Any]:
        return await self.redis.client.hgetall(f"job:{job_id}")
