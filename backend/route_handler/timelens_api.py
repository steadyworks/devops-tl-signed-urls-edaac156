import logging
import uuid
from typing import Optional

from fastapi import File, UploadFile
from pydantic import BaseModel

from backend.lib.types.asset import Asset
from backend.lib.utils.common import none_throws
from backend.lib.utils.web_requests import UploadFileTempDirManager
from backend.route_handler.base import RouteHandler


class UploadedFileInfo(BaseModel):
    filename: str
    storage_key: str


class FailedUploadInfo(BaseModel):
    filename: str
    error: str


class NewPhotobookResponse(BaseModel):
    job_id: str
    uploaded_files: list[UploadedFileInfo]
    failed_uploads: list[FailedUploadInfo]
    skipped_non_media: list[str]


class TimelensAPIHandler(RouteHandler):
    def register_routes(self) -> None:
        self.router.add_api_route(
            "/api/new_photobook",
            self.new_photobook,
            methods=["POST"],
            response_model=NewPhotobookResponse,
        )

    @staticmethod
    def is_accepted_mime(mime: Optional[str]) -> bool:
        return mime is not None and (
            mime.startswith("image/")
            # or mime.startswith("video/") # only images allowed for now
        )

    async def new_photobook(
        self, files: list[UploadFile] = File(...)
    ) -> NewPhotobookResponse:
        job_id = f"job_{uuid.uuid4().hex}"

        # Filter valid files according to FastAPI reported mime type
        valid_files = [
            file
            for file in files
            if TimelensAPIHandler.is_accepted_mime(file.content_type)
        ]
        file_names = [file.filename for file in valid_files]
        skipped = [
            file.filename
            for file in files
            if file not in valid_files and file.filename is not None
        ]
        logging.info({"accepted_files": file_names, "skipped_non_media": skipped})

        succeeded_uploads: list[UploadedFileInfo] = []
        failed_uploads: list[FailedUploadInfo] = []

        async with UploadFileTempDirManager(
            job_id, valid_files
        ) as user_requested_uploads:
            upload_results = await self.app.asset_manager.upload_files_batched(
                [
                    (
                        none_throws(asset.cached_local_path),
                        self.app.asset_manager.mint_asset_key(
                            job_id, none_throws(asset.cached_local_path).name
                        ),
                    )
                    for (_original_fname, asset) in user_requested_uploads
                ]
            )

            # Transform upload results into endpoint response
            for _original_fname, asset in user_requested_uploads:
                upload_res = upload_results.get(
                    none_throws(asset.cached_local_path), None
                )
                if upload_res is None or isinstance(upload_res, Exception):
                    failed_uploads.append(
                        FailedUploadInfo(
                            filename=_original_fname, error=str(upload_res)
                        )
                    )
                else:
                    assert isinstance(upload_res, Asset)
                    succeeded_uploads.append(
                        UploadedFileInfo(
                            filename=_original_fname,
                            storage_key=none_throws(upload_res.asset_storage_key),
                        )
                    )

        await self.app.job_manager.enqueue(
            job_id, [file.storage_key for file in succeeded_uploads]
        )

        return NewPhotobookResponse(
            job_id=job_id,
            uploaded_files=succeeded_uploads,
            failed_uploads=failed_uploads,
            skipped_non_media=skipped,
        )
