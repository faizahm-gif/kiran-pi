"""Classify one banana-leaf photo with the saved model.

With no --image argument, the Raspberry Pi camera captures the leaf first.
The leaf is segmented before classification. Every class is printed with its
confidence (softmax probability).

Examples
--------
python inference.py --device cpu --segment-mode classic
python inference.py --image leaf.jpg --device cpu --segment-mode classic
python inference.py --camera --no-preview --warmup 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from banana_deficiency.camera import capture_banana_leaf
from banana_deficiency.config import DEFAULT_SAM_WEIGHTS, PROJECT_ROOT, get_device, release_cuda
from banana_deficiency.images import open_rgb
from banana_deficiency.model import load_checkpoint
from banana_deficiency.segment import LeafSegmenter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Segment one banana leaf and predict its nutritional deficiency.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Classify this file. Omit it, or pass --camera, to capture from the Pi Camera first.",
    )
    parser.add_argument("--camera", action="store_true", help="Capture a leaf with the Raspberry Pi camera, then classify it.")
    parser.add_argument("--no-preview", action="store_true", help="Capture without a live camera preview.")
    parser.add_argument("--warmup", type=float, default=2.0, help="Seconds to let the camera settle before capture.")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "best")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--sam-weights", default=DEFAULT_SAM_WEIGHTS)
    parser.add_argument(
        "--segment-mode",
        choices=("auto", "sam", "classic"),
        default="auto",
        help="auto removes a black backdrop directly and uses GrabCut when SAM 2 is unavailable. classic never loads SAM 2.",
    )
    parser.add_argument("--no-segment", action="store_true", help="Skip background removal.")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--no-chart", action="store_true", help="Skip the confidence bar chart.")
    return parser.parse_args()


def save_chart(classes: list[str], probabilities: np.ndarray, predicted: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = ["#1b7f4e" if name == predicted else "#8aa0b2" for name in classes]
    bars = ax.bar(classes, probabilities, color=colors)
    for bar, value in zip(bars, probabilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Confidence")
    ax.set_title(f"Predicted: {predicted}")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def resolve_image(args) -> Path:
    if args.image is not None and not args.camera:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise SystemExit(f"Image not found: {image_path}")
        return image_path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = PROJECT_ROOT / "outputs" / "captures" / f"leaf_{stamp}.jpg"
    print("Opening the Raspberry Pi camera.")
    return capture_banana_leaf(image_path, preview=not args.no_preview, warmup_seconds=args.warmup)


def main():
    args = parse_args()
    image_path = resolve_image(args)

    device = get_device(args.device)
    output_dir = args.output_dir or (PROJECT_ROOT / "outputs" / "inference" / image_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_segment:
        print("WARNING: background removal is off.")
        rgb = np.asarray(open_rgb(image_path))
        method = "skipped"
        Image.fromarray(rgb).save(output_dir / "input.png")
    else:
        segmenter = LeafSegmenter(args.sam_weights, device, mode=args.segment_mode)
        result = segmenter.segment_file(image_path, make_preview=True)
        method = result.method
        rgb = result.image_rgb
        Image.fromarray(rgb).save(output_dir / "segmented_leaf.png")
        if result.preview_bgr is not None:
            preview_rgb = np.ascontiguousarray(result.preview_bgr[:, :, ::-1])
            Image.fromarray(preview_rgb).save(output_dir / "overlay.jpg", quality=90)
        del segmenter
        release_cuda()
        if method == "original":
            print("WARNING: a leaf mask was not found. The original image was used.")

    model, processor, meta = load_checkpoint(args.checkpoint, device)
    classes = list(meta["classes"])
    encoded = processor(images=Image.fromarray(rgb), return_tensors="pt")
    pixels = encoded["pixel_values"].to(device)
    with torch.inference_mode():
        logits = model(pixels)
        probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()[0]

    order = np.argsort(-probabilities)
    predicted = classes[int(order[0])]
    confidence = float(probabilities[order[0]])
    ranking = [
        {"class": classes[int(index)], "confidence": float(probabilities[index])}
        for index in order
    ]

    print(f"Image: {image_path}")
    print(f"Device: {device}")
    print(f"Background removal: {method}")
    print(f"Predicted: {predicted}")
    print(f"Confidence: {confidence:.4f} ({confidence * 100:.2f}%)")
    print()
    print(f"{'Rank':<6}{'Class':<14}{'Confidence':>12}")
    for rank, item in enumerate(ranking, start=1):
        print(
            f"{rank:<6}{item['class']:<14}{item['confidence']:12.4f}"
            f"  ({item['confidence'] * 100:6.2f}%)"
        )

    payload = {
        "image": str(image_path.resolve()),
        "device": str(device),
        "segmentation_method": method,
        "predicted_class": predicted,
        "confidence": confidence,
        "top_k": ranking[: max(1, args.topk)],
        "probabilities": {item["class"]: item["confidence"] for item in ranking},
        "segmented_leaf": str((output_dir / "segmented_leaf.png").resolve()) if not args.no_segment else None,
        "checkpoint": {
            "model_name": meta.get("model_name"),
            "epoch": meta.get("epoch"),
            "phase": meta.get("phase"),
            "val_f1_macro": meta.get("val_f1_macro"),
        },
    }
    (output_dir / "prediction.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.no_chart:
        print(f"\nSaved results to {output_dir.resolve()}")
        return
    try:
        save_chart(classes, probabilities, predicted, output_dir / "confidence.png")
    except ImportError:
        print("WARNING: matplotlib is not installed, so confidence.png was skipped.")
    print(f"\nSaved results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
