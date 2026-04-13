import os
from pathlib import Path
from inspect import signature


DEFAULT_REPO_TYPE = "dataset"


def resolve_token(token: str | None) -> str:
    resolved = token or os.environ.get("MODELSCOPE_API_TOKEN", "")
    if not resolved:
        raise ValueError(
            "ModelScope token is required. Pass --token or set MODELSCOPE_API_TOKEN."
        )
    return resolved


def resolve_optional_token(token: str | None) -> str | None:
    resolved = token or os.environ.get("MODELSCOPE_API_TOKEN", "")
    return resolved or None


def ensure_local_dir(path: str) -> Path:
    local_dir = Path(path).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir


def ensure_existing_dir(path: str) -> Path:
    local_dir = Path(path).expanduser().resolve()
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")
    if not local_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {local_dir}")
    return local_dir


def login_hub(token: str):
    try:
        from modelscope.hub.api import HubApi
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "modelscope is required. Install it with `pip install modelscope`."
        ) from exc

    api = HubApi()
    api.login(token)
    return api


def call_with_supported_kwargs(func, **kwargs):
    supported = signature(func).parameters
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return func(**filtered_kwargs)
