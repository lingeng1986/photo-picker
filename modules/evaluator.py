"""
Phase 2 evaluator: send thumbnails to local Ollama bakllava and parse structured results.
Gracefully degrades when Ollama is unreachable or the model is missing.
"""

import base64
import json
import re
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "bakllava"
DEFAULT_TIMEOUT = 60
DEFAULT_BATCH_SIZE = 4

_EVAL_PROMPT = """\
You are a professional photo editor evaluating a portrait photo.
Respond ONLY with a single JSON object — no markdown, no explanation.

JSON schema:
{
  "eyes_open": true or false,
  "expression": "natural" | "smile" | "stiff" | "forced",
  "eye_contact": "good" | "moderate" | "poor" | "none",
  "quality": "excellent" | "good" | "average" | "poor",
  "lighting": "good" | "over" | "under",
  "composition": "good" | "centered" | "off",
  "reasoning": "<one short sentence>"
}

Evaluate the image now."""


@dataclass
class AIEvaluation:
    eyes_open: bool
    expression: str       # natural / smile / stiff / forced
    eye_contact: str      # good / moderate / poor / none
    quality: str          # excellent / good / average / poor
    lighting: str         # good / over / under
    composition: str      # good / centered / off
    reasoning: str
    skipped: bool = False  # True when Ollama was unreachable or parsing failed


def _default_evaluation(reason: str = "skipped") -> AIEvaluation:
    return AIEvaluation(
        eyes_open=True,
        expression="natural",
        eye_contact="moderate",
        quality="average",
        lighting="good",
        composition="centered",
        reasoning=reason,
        skipped=True,
    )


def _encode_image(image_path: Path) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError as e:
        print(f"    [warn] Cannot read {image_path.name}: {e}")
        return None


def _call_ollama(image_b64: str, model: str, timeout: int) -> Optional[str]:
    payload = json.dumps({
        "model": model,
        "prompt": _EVAL_PROMPT,
        "images": [image_b64],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            return data.get("response", "")
    except urllib.error.URLError as e:
        print(f"    [warn] Ollama unreachable: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"    [warn] Unexpected Ollama response: {e}")
        return None


def _parse_response(text: str, image_name: str) -> AIEvaluation:
    """Extract JSON from Ollama response, tolerate markdown fences."""
    if not text:
        return _default_evaluation("empty response")

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip()

    # Find first {...} block
    match = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if not match:
        print(f"    [warn] No JSON found in response for {image_name}")
        return _default_evaluation("no JSON in response")

    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"    [warn] JSON parse error for {image_name}: {e}")
        return _default_evaluation("JSON parse error")

    def _str(key, default, valid):
        v = str(obj.get(key, default)).lower()
        return v if v in valid else default

    return AIEvaluation(
        eyes_open=bool(obj.get("eyes_open", True)),
        expression=_str("expression", "natural", {"natural", "smile", "stiff", "forced"}),
        eye_contact=_str("eye_contact", "moderate", {"good", "moderate", "poor", "none"}),
        quality=_str("quality", "average", {"excellent", "good", "average", "poor"}),
        lighting=_str("lighting", "good", {"good", "over", "under"}),
        composition=_str("composition", "centered", {"good", "centered", "off"}),
        reasoning=str(obj.get("reasoning", ""))[:200],
        skipped=False,
    )


def _check_ollama_available(model: str, timeout: int) -> bool:
    """Quick health-check: hit /api/tags and confirm model is listed."""
    try:
        req = urllib.request.Request(
            "http://localhost:11434/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            # bakllava may appear as "bakllava:latest"
            return any(m == model or m.startswith(model + ":") for m in models)
    except Exception:
        return False


class Evaluator:
    def __init__(self, config: dict):
        ai_cfg = config.get("ai_evaluation", {})
        self.model = ai_cfg.get("model", DEFAULT_MODEL)
        self.timeout = ai_cfg.get("timeout_seconds", DEFAULT_TIMEOUT)
        self.batch_size = ai_cfg.get("batch_size", DEFAULT_BATCH_SIZE)
        self._available: Optional[bool] = None

    def _is_available(self) -> bool:
        if self._available is None:
            self._available = _check_ollama_available(self.model, self.timeout)
            if not self._available:
                print(f"[eval] Model '{self.model}' not available — AI evaluation will be skipped")
        return self._available

    def evaluate_single(self, image_path: Path) -> AIEvaluation:
        if not self._is_available():
            return _default_evaluation("ollama unavailable")

        source = Path(image_path)
        if not source.exists():
            return _default_evaluation("file not found")

        b64 = _encode_image(source)
        if b64 is None:
            return _default_evaluation("cannot read image")

        raw = _call_ollama(b64, self.model, self.timeout)
        return _parse_response(raw or "", source.name)

    def evaluate_batch(self, images) -> list[AIEvaluation]:
        """
        Evaluate a list of ImageInfo objects (or paths).
        Returns one AIEvaluation per image, in the same order.
        """
        results = []
        total = len(images)
        for i, img in enumerate(images, 1):
            path = img.thumb_path if (hasattr(img, "thumb_path") and img.thumb_path and img.thumb_path.exists()) else img.path
            print(f"  [eval {i}/{total}] {Path(path).name} ...", end=" ", flush=True)
            ev = self.evaluate_single(path)
            if ev.skipped:
                print(f"skipped ({ev.reasoning})")
            else:
                eyes = "open" if ev.eyes_open else "CLOSED"
                print(f"{ev.quality} | eyes={eyes} | expr={ev.expression} | {ev.reasoning[:60]}")
            results.append(ev)
        return results
