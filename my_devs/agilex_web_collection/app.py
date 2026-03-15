from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import (
    JobDetail,
    JobSummary,
    LogReadResult,
    PreflightResponse,
    RecordRequest,
    RuntimeInfo,
    StopJobResponse,
)
from .paths import AgilexWebPaths
from .service import JobConflictError, JobNotFoundError, JobStartError, RecordJobService
from .store import FileJobStore
from .supervisor import ProcessSupervisor


LOGGER = logging.getLogger(__name__)
APP_TITLE = "agilex_web_collection"
APP_VERSION = "0.1.0"

PATHS = AgilexWebPaths.discover()
STORE = FileJobStore(PATHS)
SUPERVISOR = ProcessSupervisor()
SERVICE = RecordJobService(paths=PATHS, store=STORE, supervisor=SUPERVISOR)


def _build_lifespan(service: RecordJobService):
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.initialize()
        yield

    return lifespan


def _status_phrase(status_code: int) -> str:
    try:
        return HTTPStatus(status_code).phrase
    except ValueError:
        return "Request failed"


def _default_error_code(status_code: int) -> str:
    return {
        400: "bad_request",
        404: "not_found",
        409: "conflict",
        415: "unsupported_media_type",
        422: "validation_error",
        500: "internal_error",
        503: "service_unavailable",
    }.get(status_code, f"http_{status_code}")


def _error_payload(status_code: int, code: str, message: str, *, details: Any = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error": {
            "status_code": status_code,
            "code": code,
            "message": message,
        }
    }
    if details is not None:
        payload["error"]["details"] = details
    return payload


def _error_response(status_code: int, code: str, message: str, *, details: Any = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=jsonable_encoder(_error_payload(status_code, code, message, details=details)),
    )


def _parse_http_exception(status_code: int, detail: Any) -> tuple[str, str, Any]:
    if isinstance(detail, dict):
        code = str(detail.get("code") or _default_error_code(status_code))
        message = str(detail.get("message") or detail.get("detail") or _status_phrase(status_code))
        details = detail.get("details")
        extras = {
            key: value
            for key, value in detail.items()
            if key not in {"code", "message", "detail", "details"}
        }
        if extras:
            if details is None:
                details = extras
            elif isinstance(details, dict):
                details = {**details, **extras}
            else:
                details = {"details": details, **extras}
        return code, message, details

    if isinstance(detail, str):
        return _default_error_code(status_code), detail, None

    if detail is None:
        return _default_error_code(status_code), _status_phrase(status_code), None

    return _default_error_code(status_code), _status_phrase(status_code), detail


def _is_json_media_type(content_type: str | None) -> bool:
    if not content_type:
        return False
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


async def require_json_request(request: Request) -> None:
    if _is_json_media_type(request.headers.get("content-type")):
        return
    raise HTTPException(
        status_code=415,
        detail={
            "code": "unsupported_media_type",
            "message": "This endpoint only accepts structured JSON request bodies.",
            "details": {"content_type": request.headers.get("content-type")},
        },
    )


async def _http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    code, message, details = _parse_http_exception(exc.status_code, exc.detail)
    return _error_response(exc.status_code, code, message, details=details)


async def _validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    message = "Request validation failed."
    if any(error.get("type") == "json_invalid" for error in exc.errors()):
        message = "Request body is not valid JSON."
    return _error_response(
        status_code=422,
        code="validation_error",
        message=message,
        details={"errors": exc.errors()},
    )


async def _job_not_found_handler(_: Request, exc: JobNotFoundError) -> JSONResponse:
    job_id = exc.args[0] if exc.args else ""
    return _error_response(
        status_code=404,
        code="job_not_found",
        message=f"Unknown job_id: {job_id}.",
        details={"job_id": job_id},
    )


async def _job_conflict_handler(_: Request, exc: JobConflictError) -> JSONResponse:
    return _error_response(
        status_code=409,
        code="job_conflict",
        message=str(exc),
        details={
            "conflicts": exc.conflicts,
            "preflight": exc.preflight.model_dump(mode="json"),
        },
    )


async def _job_start_handler(_: Request, exc: JobStartError) -> JSONResponse:
    return _error_response(
        status_code=500,
        code="job_start_error",
        message="Failed to launch record.sh.",
        details={"reason": str(exc)},
    )


async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled exception on %s %s", request.method, request.url.path, exc_info=exc)
    return _error_response(status_code=500, code="internal_error", message="Unhandled server error.")


def create_app(
    *,
    paths: AgilexWebPaths = PATHS,
    service: RecordJobService = SERVICE,
) -> FastAPI:
    app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=_build_lifespan(service))
    app.state.paths = paths
    app.state.service = service
    app.mount("/static", StaticFiles(directory=str(paths.static_dir)), name="static")

    app.add_exception_handler(HTTPException, _http_exception_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(RequestValidationError, _validation_exception_handler)
    app.add_exception_handler(JobNotFoundError, _job_not_found_handler)
    app.add_exception_handler(JobConflictError, _job_conflict_handler)
    app.add_exception_handler(JobStartError, _job_start_handler)
    app.add_exception_handler(Exception, _unhandled_exception_handler)

    @app.get("/")
    def index() -> FileResponse:
        index_path = paths.static_dir / "index.html"
        if not index_path.is_file():
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "static_homepage_missing",
                    "message": "Static homepage is not available.",
                    "details": {"path": str(index_path)},
                },
            )
        return FileResponse(str(index_path))

    @app.get("/api/runtime", response_model=RuntimeInfo)
    def get_runtime() -> RuntimeInfo:
        return service.runtime_info()

    @app.post("/api/preflight", response_model=PreflightResponse)
    def run_preflight(
        payload: RecordRequest,
        _: None = Depends(require_json_request),
    ) -> PreflightResponse:
        return service.preflight(payload)

    @app.post("/api/jobs", response_model=JobDetail, status_code=201)
    def create_job(
        payload: RecordRequest,
        _: None = Depends(require_json_request),
    ) -> JobDetail:
        return service.create_job(payload)

    @app.get("/api/jobs/active", response_model=JobDetail | None)
    def get_active_job() -> JobDetail | None:
        return service.get_active_job()

    @app.get("/api/jobs/recent", response_model=list[JobSummary])
    def list_recent_jobs(limit: int = Query(default=20, ge=1, le=100)) -> list[JobSummary]:
        return service.list_jobs(limit=limit)

    @app.get("/api/jobs", response_model=list[JobSummary], include_in_schema=False)
    def list_jobs_compat(limit: int = Query(default=20, ge=1, le=100)) -> list[JobSummary]:
        return service.list_jobs(limit=limit)

    @app.get("/api/jobs/{job_id}", response_model=JobDetail)
    def get_job(job_id: str) -> JobDetail:
        return service.get_job(job_id)

    @app.get("/api/jobs/{job_id}/logs", response_model=LogReadResult)
    def get_job_logs(
        job_id: str,
        cursor: int = Query(default=0, ge=0),
        limit_bytes: int = Query(default=65536, ge=1024, le=1048576),
    ) -> LogReadResult:
        return service.get_logs(job_id, cursor=cursor, limit_bytes=limit_bytes)

    @app.post("/api/jobs/{job_id}/stop", response_model=StopJobResponse)
    def stop_job(job_id: str) -> StopJobResponse:
        return service.stop_job(job_id)

    return app


app = create_app()

