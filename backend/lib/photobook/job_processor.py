import asyncio
import tempfile
from pathlib import Path
from typing import Any

from backend.lib.asset_manager.base import AssetManager
from backend.lib.types.asset import Asset
from backend.lib.utils.common import none_throws
from backend.lib.vertex_ai.gemini import Gemini


class JobProcessor:
    def __init__(
        self, job_id: str, job_data: dict[str, Any], asset_manager: AssetManager
    ) -> None:
        self.job_id = job_id
        self.job_data = job_data
        self.image_keys: list[str] = job_data.get("image_keys", [])
        self.instruction: str = job_data.get("instruction", "")
        self.asset_manager = asset_manager
        self.gemini = Gemini()

        self._image_download_semaphore = asyncio.Semaphore(10)

    async def _download_all_images(self, tmpdir_path: Path) -> list[Path]:
        download_results = await self.asset_manager.download_files_batched(
            [(key, tmpdir_path / Path(key).name) for key in self.image_keys]
        )
        return [
            none_throws(asset.cached_local_path)
            for asset in download_results.values()
            if isinstance(asset, Asset)
        ]

    async def process(self) -> dict[str, Any]:
        if not self.image_keys:
            raise ValueError("No image_keys found in job_data")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Download
            tmp_path = Path(tmpdir)
            downloaded_paths = await self._download_all_images(tmp_path)
            if not downloaded_paths:
                raise RuntimeError("All image downloads failed")

            # Run gemini
            try:
                gemini_output = await self.gemini.run_image_understanding_job(
                    self.instruction, downloaded_paths
                )
            except Exception as e:
                gemini_output = f"Gemini generation failed: {e}"

        return {
            "job_id": self.job_id,
            "processed_keys": self.image_keys,
            "successful_files": [str(p) for p in downloaded_paths],
            "gemini_result": gemini_output,
        }
