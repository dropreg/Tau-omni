from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentBundle:
    text: str
    media_blocks: list[dict[str, Any]] = field(default_factory=list)


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_media_block(part: dict[str, Any]) -> dict[str, Any] | None:
    block_type = part.get("type")

    if block_type == "image_url":
        image_url = part.get("image_url")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, dict) and image_url.get("url"):
            return {"type": "image_url", "image_url": {"url": image_url["url"]}}

    if block_type == "video_url":
        video_url = part.get("video_url")
        if isinstance(video_url, str):
            return {"type": "video_url", "video_url": {"url": video_url}}
        if isinstance(video_url, dict) and video_url.get("url"):
            return {"type": "video_url", "video_url": {"url": video_url["url"]}}

    if block_type == "image" and part.get("image"):
        return {"type": "image_url", "image_url": {"url": part["image"]}}

    if block_type == "video" and part.get("video"):
        return {"type": "video_url", "video_url": {"url": part["video"]}}

    return None


def normalize_content(content: Any) -> ContentBundle:
    if isinstance(content, str):
        return ContentBundle(text=content)

    if isinstance(content, list):
        text_parts: list[str] = []
        media_blocks: list[dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                text_parts.append(_as_text(part))
                continue

            block_type = part.get("type")
            if block_type == "text":
                text_parts.append(_as_text(part.get("text", part.get("content", ""))))
                continue

            media_block = _normalize_media_block(part)
            if media_block is not None:
                media_blocks.append(media_block)
                continue

            if "content" in part:
                text_parts.append(_as_text(part["content"]))

        return ContentBundle(text="\n".join(p for p in text_parts if p), media_blocks=media_blocks)

    return ContentBundle(text=_as_text(content))


def _append_media_section(
    content_blocks: list[dict[str, Any]],
    title: str,
    media_blocks: list[dict[str, Any]],
) -> None:
    if not media_blocks:
        return
    content_blocks.append({"type": "text", "text": title})
    content_blocks.extend(media_blocks)


def build_user_message_content(
    prompt_text: str,
    query_bundle: ContentBundle,
    response_a_bundle: ContentBundle,
    response_b_bundle: ContentBundle,
):
    media_count = (
        len(query_bundle.media_blocks)
        + len(response_a_bundle.media_blocks)
        + len(response_b_bundle.media_blocks)
    )
    if media_count == 0:
        return prompt_text

    content_blocks: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    _append_media_section(content_blocks, "[Query Media]", query_bundle.media_blocks)
    _append_media_section(content_blocks, "[Assistant A Media]", response_a_bundle.media_blocks)
    _append_media_section(content_blocks, "[Assistant B Media]", response_b_bundle.media_blocks)
    return content_blocks
