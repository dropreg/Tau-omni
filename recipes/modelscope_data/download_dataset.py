import argparse

from common import (
    DEFAULT_REPO_TYPE,
    call_with_supported_kwargs,
    ensure_local_dir,
    login_hub,
    resolve_optional_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a dataset repository from ModelScope to a local directory."
    )
    parser.add_argument(
        "--dataset-id",
        required=True,
        help="ModelScope dataset id, for example `org_name/dataset_name`.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory used to store downloaded dataset files.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision, branch, or tag.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional ModelScope token. Falls back to MODELSCOPE_API_TOKEN.",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        help="Repository type. Default is `dataset`.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = resolve_optional_token(args.token)
    local_dir = ensure_local_dir(args.local_dir)
    if token:
        login_hub(token)

    try:
        from modelscope import snapshot_download
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "modelscope is required. Install it with `pip install modelscope`."
        ) from exc

    call_with_supported_kwargs(
        snapshot_download,
        repo_id=args.dataset_id,
        dataset_id=args.dataset_id,
        repo_type=args.repo_type,
        revision=args.revision,
        cache_dir=str(local_dir),
        local_dir=str(local_dir),
    )

    print(f"Downloaded {args.dataset_id} to {local_dir}")


if __name__ == "__main__":
    main()
