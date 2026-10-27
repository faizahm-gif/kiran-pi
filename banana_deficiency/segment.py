"""Remove everything except the banana leaf.

Auto mode:
- A photo is treated as a leaf on a black backdrop when near-black pixels
  touch the frame and almost everything else is green or yellow. The mask
  keeps every pixel that is not that backdrop, including brown or dark
  deficiency spots. Enclosed dark spots are filled back in.
- Photos with soil, sky, hands, cloth, or a bench are segmented with SAM 2.
  Positive prompts come from the green and yellow leaf region; corners are
  negative prompts.
- If SAM 2 is missing or rejects the mask, GrabCut and a color prior are
  used so a leaf is still isolated.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from banana_deficiency.images import open_bgr

BLACK_THRESH = 12


@dataclass
class SegmentResult:
    image_rgb: np.ndarray
    mask: np.ndarray
    method: str
    preview_bgr: np.ndarray | None = None


class SamUnavailable(RuntimeError):
    pass


def _filter_components(mask: np.ndarray, min_frac: float) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    height, width = mask.shape
    minimum = min_frac * height * width
    kept = np.zeros_like(mask)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            kept[labels == index] = 255
    if kept.max() == 0:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        kept[labels == largest] = 255
    return kept


def _fill_small_holes(mask: np.ndarray, max_frac: float) -> np.ndarray:
    height, width = mask.shape
    max_area = max(1, int(max_frac * height * width))
    inverse = np.where(mask > 0, 0, 255).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    filled = mask.copy()
    for index in range(1, count):
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        box_w = int(stats[index, cv2.CC_STAT_WIDTH])
        box_h = int(stats[index, cv2.CC_STAT_HEIGHT])
        touches_border = x == 0 or y == 0 or (x + box_w) >= width or (y + box_h) >= height
        if touches_border:
            continue
        if int(stats[index, cv2.CC_STAT_AREA]) <= max_area:
            filled[labels == index] = 255
    return filled


def _close(mask: np.ndarray, size: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def is_leaf_on_black(bgr: np.ndarray, thresh: int = BLACK_THRESH) -> bool:
    """True when the background is black and the foreground is the leaf.

    A leaf that touches the frame still counts. A scene with soil, sky, or
    hands does not, even if a corner is dark, because those pixels are not
    near-black and are not leaf-colored.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    black = gray <= thresh
    if float(black.mean()) < 0.015:
        return False
    touches_frame = bool(black[0].any() or black[-1].any() or black[:, 0].any() or black[:, -1].any())
    if not touches_frame:
        return False
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    leaf_color = color_prior(bgr) > 0
    # Pale or white leaf tissue is unsaturated. Saturated leftovers are soil, sky, or cloth.
    scene = (~black) & (~leaf_color) & (saturation >= 45)
    return float(scene.mean()) <= 0.12


def border_black_fraction(bgr: np.ndarray, thresh: int = BLACK_THRESH) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    margin = max(2, int(round(0.03 * min(height, width))))
    strips = (
        gray[:margin, :].ravel(),
        gray[-margin:, :].ravel(),
        gray[:, :margin].ravel(),
        gray[:, -margin:].ravel(),
    )
    border = np.concatenate(strips)
    return float((border <= thresh).mean())


def mask_black_background(bgr: np.ndarray, thresh: int = BLACK_THRESH) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = np.where(gray > thresh, 255, 0).astype(np.uint8)
    mask = _close(mask, 5)
    mask = _filter_components(mask, min_frac=0.01)
    return _fill_small_holes(mask, max_frac=0.003)


def color_prior(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    green = (hue >= 20) & (hue <= 100) & (sat >= 25) & (val >= 20)
    yellow = (hue >= 5) & (hue <= 40) & (sat >= 30) & (val >= 35)
    prior = np.where(green | yellow, 255, 0).astype(np.uint8)
    prior = _close(prior, 7)
    small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    prior = cv2.morphologyEx(prior, cv2.MORPH_OPEN, small)
    return _filter_components(prior, min_frac=0.01)


def _resize_max(image: np.ndarray, max_side: int, interpolation: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 0.999:
        return image, 1.0
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=interpolation,
    )
    return resized, scale


def grabcut_mask(bgr: np.ndarray, prior: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    small, scale = _resize_max(bgr, 800, cv2.INTER_AREA)
    if scale < 0.999:
        prior_small = cv2.resize(prior, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        prior_small = prior
    work_h, work_w = small.shape[:2]
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        if int(prior_small.sum()) == 0:
            rect = (
                int(work_w * 0.08),
                int(work_h * 0.08),
                max(1, int(work_w * 0.84)),
                max(1, int(work_h * 0.84)),
            )
            gc_mask = np.zeros((work_h, work_w), np.uint8)
            cv2.grabCut(small, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        else:
            gc_mask = np.full((work_h, work_w), cv2.GC_PR_BGD, np.uint8)
            sure_fg = cv2.erode(prior_small, np.ones((5, 5), np.uint8))
            dilated = cv2.dilate(prior_small, np.ones((21, 21), np.uint8))
            gc_mask[prior_small > 0] = cv2.GC_PR_FGD
            gc_mask[sure_fg > 0] = cv2.GC_FGD
            gc_mask[dilated == 0] = cv2.GC_BGD
            cv2.grabCut(small, gc_mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return prior
    foreground = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    if scale < 0.999:
        foreground = cv2.resize(foreground, (width, height), interpolation=cv2.INTER_NEAREST)
    return foreground


def _overlap(mask: np.ndarray, prior: np.ndarray) -> tuple[float, float, float]:
    leaf = mask > 0
    hint = prior > 0
    fraction = float(leaf.mean())
    if int(hint.sum()) < 10:
        return 1.0, 0.0, fraction
    intersection = np.logical_and(leaf, hint).sum()
    cover = float(intersection) / float(hint.sum())
    extra = float(np.logical_and(leaf, ~hint).sum()) / float(max(int(leaf.sum()), 1))
    return cover, extra, fraction


def _acceptable(mask: np.ndarray, prior: np.ndarray) -> bool:
    cover, extra, fraction = _overlap(mask, prior)
    if fraction < 0.03 or fraction > 0.98:
        return False
    if int((prior > 0).sum()) < 10:
        return 0.08 <= fraction <= 0.92
    return cover >= 0.60 and extra <= 0.55


def _score(mask: np.ndarray, prior: np.ndarray) -> float:
    cover, extra, fraction = _overlap(mask, prior)
    if fraction < 0.03 or fraction > 0.98:
        return -1.0
    if int((prior > 0).sum()) < 10:
        return fraction
    return 0.70 * cover - 0.35 * extra + 0.10 * min(fraction, 0.80)


def build_prompts(prior: np.ndarray) -> tuple[list[list[int]], list[int], list[int] | None]:
    height, width = prior.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats((prior > 0).astype(np.uint8), 8)
    raw_points: list[tuple[int, int, int]] = []
    bbox = None
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, box_w, box_h, _ = (int(v) for v in stats[largest])
        component = np.where(labels == largest, 255, 0).astype(np.uint8)
        distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        flat = int(np.argmax(distance))
        peak_y, peak_x = divmod(flat, width)
        raw_points.append((int(peak_x), int(peak_y), 1))
        ys, xs = np.where(distance > max(2.0, 0.35 * float(distance.max())))
        if len(xs):
            order = np.argsort(ys * width + xs)
            picks = np.linspace(0, len(order) - 1, num=min(3, len(order)), dtype=int)
            for pick in picks:
                raw_points.append((int(xs[order[pick]]), int(ys[order[pick]]), 1))
        pad_x = int(0.08 * box_w)
        pad_y = int(0.08 * box_h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width - 1, x + box_w + pad_x)
        y2 = min(height - 1, y + box_h + pad_y)
        if (x2 - x1) * (y2 - y1) < 0.92 * height * width:
            bbox = [x1, y1, x2, y2]
        _ = centroids
    else:
        raw_points.append((width // 2, height // 2, 1))
        bbox = [int(width * 0.10), int(height * 0.10), int(width * 0.90), int(height * 0.90)]

    corners = (
        (2, 2),
        (width - 3, 2),
        (2, height - 3),
        (width - 3, height - 3),
        (width // 2, 2),
        (width // 2, height - 3),
        (2, height // 2),
        (width - 3, height // 2),
    )
    negatives = 0
    for x, y in corners:
        x = int(np.clip(x, 0, width - 1))
        y = int(np.clip(y, 0, height - 1))
        if prior[y, x] == 0:
            raw_points.append((x, y, 0))
            negatives += 1
        if negatives >= 4:
            break

    seen: set[tuple[int, int]] = set()
    positives: list[list[int]] = []
    pos_labels: list[int] = []
    negatives_xy: list[list[int]] = []
    neg_labels: list[int] = []
    for x, y, label in raw_points:
        key = (int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1)))
        if key in seen:
            continue
        seen.add(key)
        if label == 1 and len(positives) < 4:
            positives.append([key[0], key[1]])
            pos_labels.append(1)
        elif label == 0 and len(negatives_xy) < 4:
            negatives_xy.append([key[0], key[1]])
            neg_labels.append(0)
    if not positives:
        positives = [[width // 2, height // 2]]
        pos_labels = [1]
    points = positives + negatives_xy
    labels_out = pos_labels + neg_labels
    return points, labels_out, bbox


def _clean_object_mask(mask: np.ndarray) -> np.ndarray:
    mask = _close(mask, 5)
    mask = _filter_components(mask, min_frac=0.01)
    return _fill_small_holes(mask, max_frac=0.005)


def apply_mask_and_crop(bgr: np.ndarray, mask: np.ndarray, pad_frac: float = 0.03) -> np.ndarray:
    if mask.shape != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    foreground = bgr.copy()
    foreground[mask == 0] = 0
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = mask.shape
    pad = int(round(pad_frac * max(height, width)))
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(width, int(xs.max()) + pad + 1)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(height, int(ys.max()) + pad + 1)
    if (x2 - x1) < 32 or (y2 - y1) < 32:
        crop = foreground
    else:
        crop = foreground[y1:y2, x1:x2]
    return cv2.cvtColor(np.ascontiguousarray(crop), cv2.COLOR_BGR2RGB)


def build_preview(bgr: np.ndarray, mask: np.ndarray, cropped_rgb: np.ndarray, method: str) -> np.ndarray:
    vis = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, int(round(min(bgr.shape[:2]) / 200)))
    cv2.drawContours(vis, contours, -1, (0, 255, 0), thickness)
    cv2.putText(
        vis,
        method,
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    crop = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)

    def fit(image: np.ndarray, height: int = 512) -> np.ndarray:
        scale = height / max(image.shape[0], 1)
        size = (max(1, int(round(image.shape[1] * scale))), height)
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    return np.concatenate([fit(vis), fit(crop)], axis=1)


class LeafSegmenter:
    def __init__(self, sam_weights: str, device: torch.device, mode: str = "auto", max_side: int = 1024):
        if mode not in {"auto", "sam", "classic"}:
            raise ValueError(f"Unknown segment mode: {mode}")
        self.sam_weights = sam_weights
        self.device = device
        self.mode = mode
        self.max_side = max_side
        self._sam = None
        self._sam_error: str | None = None
        self._warned = False
        if device.type == "cuda":
            index = 0 if device.index is None else device.index
            self.ultra_device = str(index)
        else:
            self.ultra_device = "cpu"

    def segment_file(self, path, make_preview: bool = False) -> SegmentResult:
        bgr = open_bgr(path)
        if bgr is None or bgr.size == 0:
            raise FileNotFoundError(path)
        return self.segment(bgr, make_preview=make_preview)

    def segment(self, bgr: np.ndarray, make_preview: bool = False) -> SegmentResult:
        height, width = bgr.shape[:2]
        if min(height, width) < 8:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return SegmentResult(rgb, np.full((height, width), 255, np.uint8), "original", None)

        if self.mode == "classic":
            mask, method = self._classic(bgr)
        elif self.mode == "sam":
            mask, method = self._sam_or_fallback(bgr)
        else:
            if is_leaf_on_black(bgr):
                mask = mask_black_background(bgr)
                method = "black_background"
                if float((mask > 0).mean()) < 0.02:
                    mask, method = self._sam_or_fallback(bgr)
            else:
                mask, method = self._sam_or_fallback(bgr)

        mask = _fill_small_holes(mask, max_frac=0.20)
        mask = np.ascontiguousarray(mask.astype(np.uint8))
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        rgb = apply_mask_and_crop(bgr, mask)
        preview = build_preview(bgr, mask, rgb, method) if make_preview else None
        return SegmentResult(np.ascontiguousarray(rgb), mask, method, preview)

    def _classic(self, bgr: np.ndarray) -> tuple[np.ndarray, str]:
        if is_leaf_on_black(bgr):
            mask = mask_black_background(bgr)
            if float((mask > 0).mean()) >= 0.02:
                return mask, "black_background"
        prior = color_prior(bgr)
        cut = _clean_object_mask(grabcut_mask(bgr, prior))
        if _acceptable(cut, prior):
            return cut, "grabcut"
        if int(prior.sum()) > 0 and float((prior > 0).mean()) >= 0.02:
            return prior, "color_prior"
        if float((cut > 0).mean()) >= 0.02:
            return cut, "grabcut"
        return np.full(bgr.shape[:2], 255, np.uint8), "original"

    def _sam_or_fallback(self, bgr: np.ndarray) -> tuple[np.ndarray, str]:
        prior = color_prior(bgr)
        chosen = self._run_sam(bgr, prior)
        if chosen is not None:
            return chosen, "sam2"
        return self._classic(bgr)

    def _load_sam(self):
        if self._sam is not None:
            return self._sam
        if self._sam_error is not None:
            raise SamUnavailable(self._sam_error)
        try:
            from ultralytics import SAM

            self._sam = SAM(self.sam_weights)
        except Exception as exc:
            self._sam_error = str(exc)
            raise SamUnavailable(str(exc)) from exc
        return self._sam

    def _run_sam(self, bgr: np.ndarray, prior: np.ndarray) -> np.ndarray | None:
        try:
            self._load_sam()
        except SamUnavailable as exc:
            if not self._warned:
                print("WARNING: SAM 2 is unavailable. Using GrabCut / color segmentation instead.")
                print(exc)
                self._warned = True
            return None

        small, scale = _resize_max(bgr, self.max_side, cv2.INTER_AREA)
        if scale < 0.999:
            prior_small = cv2.resize(prior, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            prior_small = prior
        points, labels, bbox = build_prompts(prior_small)
        try:
            masks = self._predict(small, points, labels, bbox)
        except Exception as exc:
            if not self._warned:
                print(f"WARNING: SAM 2 failed ({exc}). Using GrabCut / color segmentation instead.")
                self._warned = True
            return None
        if not masks:
            return None

        best_mask = None
        best_score = -1.0
        for mask in masks:
            if mask.shape != bgr.shape[:2]:
                mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = _clean_object_mask(mask)
            score = _score(mask, prior)
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask is None or not _acceptable(best_mask, prior):
            return None
        return best_mask

    def _predict(self, image: np.ndarray, points: list[list[int]], labels: list[int], bbox: list[int] | None):
        attempts: list[dict] = []
        if bbox is not None:
            attempts.append({"points": points, "labels": labels, "bboxes": bbox})
            attempts.append({"points": [points], "labels": [labels], "bboxes": [bbox]})
        attempts.append({"points": points, "labels": labels})
        attempts.append({"points": [points], "labels": [labels]})
        if bbox is not None:
            attempts.append({"bboxes": bbox})
            attempts.append({"bboxes": [bbox]})

        target_hw = image.shape[:2]
        last_error: Exception | None = None
        for prompt in attempts:
            kwargs = {
                "verbose": False,
                "save": False,
                "imgsz": self.max_side,
                "device": self.ultra_device,
                "retina_masks": True,
            }
            kwargs.update(prompt)
            try:
                results = self._sam.predict(source=image, **kwargs)
            except TypeError as exc:
                last_error = exc
                kwargs.pop("device", None)
                kwargs.pop("retina_masks", None)
                try:
                    results = self._sam.predict(source=image, **kwargs)
                except TypeError as inner:
                    last_error = inner
                    continue
            masks = self._masks_from_results(results, target_hw)
            if masks:
                return masks
        if last_error is not None and not attempts:
            raise last_error
        return []

    @staticmethod
    def _masks_from_results(results, target_hw: tuple[int, int]) -> list[np.ndarray]:
        if not results:
            return []
        result = results[0]
        masks = getattr(result, "masks", None)
        if masks is None or getattr(masks, "data", None) is None:
            return []
        data = masks.data.detach().float().cpu().numpy()
        height, width = target_hw
        output = []
        for layer in data:
            if layer.shape != (height, width):
                layer = cv2.resize(layer, (width, height), interpolation=cv2.INTER_LINEAR)
            output.append(np.where(layer > 0.5, 255, 0).astype(np.uint8))
        return output
