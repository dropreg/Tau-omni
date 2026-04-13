import argparse

from common import (
    DEFAULT_REPO_TYPE,
    call_with_supported_kwargs,
    ensure_existing_dir,
    login_hub,
    resolve_token,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a local dataset folder to a ModelScope dataset repository."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target ModelScope repo id, for example `org_name/dataset_name`.",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local dataset directory to upload.",
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
    parser.add_argument(
        "--commit-message",
        default="upload dataset",
        help="Commit message used for this upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)
    local_dir = ensure_existing_dir(args.local_dir)
    api = login_hub(token)

    call_with_supported_kwargs(
        api.upload_folder,
        repo_id=args.repo_id,
        folder_path=str(local_dir),
        repo_type=args.repo_type,
        commit_message=args.commit_message,
        path_in_repo="",
    )

    print(f"Uploaded {local_dir} to {args.repo_id}")


if __name__ == "__main__":
    main()
