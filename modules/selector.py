"""
Phase 2 selector: score images using preprocessor + AI evaluation results,
then pick the best from each burst group.
"""

from pathlib import Path
from typing import Optional

try:
    from .preprocessor import ImageInfo, group_bursts as _group_bursts
    from .evaluator import AIEvaluation
except ImportError:
    from preprocessor import ImageInfo, group_bursts as _group_bursts
    from evaluator import AIEvaluation


def _score_image(
    img: ImageInfo,
    ev: AIEvaluation,
    weights: dict,
    is_group: bool,
) -> float:
    """
    Compute a weighted score for a single image.
    Higher = better. Returns 0.0 if the image is blurry.
    """
    if img.is_blurry:
        return 0.0

    w_eyes = weights.get("eyes_open", 10)
    w_expr = weights.get("expression_natural", 8)
    w_contact = weights.get("eye_contact", 6)
    w_quality = weights.get("quality", 5)
    w_light = weights.get("lighting", 3)
    w_comp = weights.get("composition", 2)

    score = 0.0

    # Eyes open — most critical for portraits
    if ev.eyes_open:
        score += w_eyes
    else:
        score -= w_eyes * 2  # heavy penalty for closed eyes

    # Expression
    expr_scores = {"smile": 1.0, "natural": 0.9, "stiff": 0.4, "forced": 0.2}
    score += w_expr * expr_scores.get(ev.expression, 0.5)

    # Eye contact (reduced weight for group shots — harder to achieve)
    contact_scores = {"good": 1.0, "moderate": 0.6, "poor": 0.2, "none": 0.0}
    contact_w = w_contact * (0.6 if is_group else 1.0)
    score += contact_w * contact_scores.get(ev.eye_contact, 0.3)

    # Overall quality
    quality_scores = {"excellent": 1.0, "good": 0.75, "average": 0.4, "poor": 0.1}
    score += w_quality * quality_scores.get(ev.quality, 0.4)

    # Lighting
    light_scores = {"good": 1.0, "over": 0.3, "under": 0.4}
    score += w_light * light_scores.get(ev.lighting, 0.5)

    # Composition
    comp_scores = {"good": 1.0, "centered": 0.7, "off": 0.2}
    score += w_comp * comp_scores.get(ev.composition, 0.5)

    # Blur score bonus (normalized — higher blur score = sharper)
    if img.blur_score > 0:
        blur_bonus = min(img.blur_score / 500.0, 1.0)
        score += blur_bonus

    return round(score, 3)


class Selector:
    def __init__(self, config: dict):
        sel_cfg = config.get("selection", {})
        self.weights = sel_cfg.get("weights", {})

    def score_all(
        self,
        images: list[ImageInfo],
        evaluations: list[AIEvaluation],
    ) -> list[float]:
        """Return a score for every image in the same order."""
        scores = []
        for img, ev in zip(images, evaluations):
            is_group = img.portrait_class == "group"
            scores.append(_score_image(img, ev, self.weights, is_group))
        return scores

    def select_best_in_group(
        self,
        images: list[ImageInfo],
        evaluations: list[AIEvaluation],
    ) -> Optional[ImageInfo]:
        """Pick the highest-scoring image from a burst group. Returns None if all score <= 0."""
        if not images:
            return None

        scores = self.score_all(images, evaluations)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]

        if best_score <= 0:
            return None
        return images[best_idx]

    def filter_by_criteria(
        self,
        images: list[ImageInfo],
        evaluations: list[AIEvaluation],
    ) -> tuple[list[ImageInfo], list[AIEvaluation]]:
        """
        Return images that pass both quality gates:
        - not blurry
        - AI quality in {good, excellent}
        """
        good_qualities = {"good", "excellent"}
        pairs = [
            (img, ev) for img, ev in zip(images, evaluations)
            if not img.is_blurry and ev.quality in good_qualities
        ]
        filtered = len(images) - len(pairs)
        print(f"[filter] {len(pairs)} passed criteria, {filtered} rejected")
        if not pairs:
            return [], []
        imgs, evs = zip(*pairs)
        return list(imgs), list(evs)

    def select_from_all(
        self,
        images: list[ImageInfo],
        evaluations: list[AIEvaluation],
    ) -> tuple[list[ImageInfo], list[ImageInfo]]:
        """
        Partition images into (selected, not_selected).

        Flow:
        1. filter_by_criteria — keep non-blurry + quality good/excellent
        2. group_bursts — cluster filtered images by timestamp
        3. select best per burst group; accept all filtered singletons

        Returns (selected, not_selected).
        """
        # Step 1: filter
        filtered_imgs, filtered_evs = self.filter_by_criteria(images, evaluations)
        if not filtered_imgs:
            return [], list(images)

        # Step 2: group bursts on filtered images only
        _group_bursts(filtered_imgs)

        # Step 3: score and select best per group
        scores = self.score_all(filtered_imgs, filtered_evs)

        burst_groups: dict[int, list[int]] = {}
        for i, img in enumerate(filtered_imgs):
            if img.burst_group >= 0:
                burst_groups.setdefault(img.burst_group, []).append(i)

        selected_indices: set[int] = set()

        for indices in burst_groups.values():
            best_i = max(indices, key=lambda i: scores[i])
            selected_indices.add(best_i)

        for i, img in enumerate(filtered_imgs):
            if img.burst_group == -1:
                selected_indices.add(i)

        selected = [filtered_imgs[i] for i in range(len(filtered_imgs)) if i in selected_indices]
        selected_paths = {img.path for img in selected}
        not_selected = [img for img in images if img.path not in selected_paths]

        return selected, not_selected
