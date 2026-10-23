from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def create_annotated_preview(
    source_path: Path,
    output_path: Path,
    regions: list[dict[str, Any]],
) -> None:
    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    draw = ImageDraw.Draw(image)

    for region in regions:
        x1, y1, x2, y2 = region["bbox"]

        px = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )

        label = (
            "full image fallback"
            if region.get("is_fallback")
            else f'{region["label"]} {region["confidence"]:.2f}'
        )

        draw.rectangle(
            px,
            outline="white",
            width=max(2, width // 350),
        )

        tx = px[0]
        ty = max(0, px[1] - 19)
        label_width = max(110, len(label) * 7)

        draw.rectangle(
            (tx, ty, tx + label_width, ty + 19),
            fill="black",
        )
        draw.text(
            (tx + 4, ty + 3),
            label,
            fill="white",
        )

    image.save(output_path, quality=92)
