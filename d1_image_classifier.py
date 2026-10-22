"""
D1-Omni Hardware-Accelerated Illustration Art Classifier
-------------------------------------------------------
Classifies whether an image is illustration art (anime, manga, digital art,
drawings, paintings, character art) or not (desktop screenshots, UI, photos, diagrams).

Hardware Target: Intel Iris Xe Graphics (96 EUs) / CPU via Liquid AI d1-omni-600M.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

from PIL import Image
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast
import openvino as ov

CACHE_DIR = os.path.expanduser("~/.cache/openvino_iris_xe")
os.makedirs(CACHE_DIR, exist_ok=True)

MODEL_ID = "LiquidAI/d1-omni-600M"

CLASSIFIER_QUESTIONS = {
    "medium": {
        "type": "choice",
        "instructions": "What is the primary medium and format of this image?",
        "criteria": {
            "illustration_art": "Hand-drawn artwork, anime, manga, cartoon, digital illustration, fanart, character art, sketch, painting",
            "screenshot": "Computer desktop, software user interface, browser window, code editor, application dialog, screen capture",
            "photograph": "Real-world photograph of a physical person, place, or physical object",
            "diagram": "Charts, schematics, graphs, blueprints, slides, pure text"
        }
    }
}


class D1ImageClassifier:
    """
    Illustration Art Classifier using Liquid AI's d1-omni-600M multimodal decision engine.
    """
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "CPU",
        max_image_dim: int = 720,
        engine: Any = None,
    ):
        self.model_id = model_id
        self.device = device.upper()
        self.max_image_dim = max_image_dim

        if engine is not None:
            print("[D1Classifier] Attaching to existing shared engine model...")
            self.model = getattr(engine, "model", engine)
            self.tokenizer = getattr(engine, "tokenizer", getattr(self.model, "tokenizer", None))
            self.model.vision = self.model.vision.float()
            self.model.head = self.model.head.float()
            return

        print(f"[D1Classifier] Initializing D1-Omni Image Classifier on {self.device}...")
        
        # 1. Tokenizer
        tok_path = hf_hub_download(self.model_id, "tokenizer.json")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tok_path,
            bos_token="<|startoftext|>",
            eos_token="<|im_end|>",
            pad_token="<|pad|>",
            mask_token="<|mask|>"
        )

        # 2. Config & Pre-trained weights
        t0 = time.perf_counter()
        self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self.config.max_length = 16384

        self.model = AutoModel.from_pretrained(
            self.model_id,
            config=self.config,
            trust_remote_code=True,
            dtype=torch.float32,
        )
        self.model.eval()
        self.model.config.max_length = 16384
        self.model.name_or_path = self.model_id
        self.model.__dict__["tokenizer"] = self.tokenizer
        self.model.vision = self.model.vision.float()
        self.model.head = self.model.head.float()

        print(f"[D1Classifier] Model initialized in {time.perf_counter() - t0:.2f}s")

    def preprocess_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Loads and prepares an image. Downscales images exceeding max_image_dim
        to preserve high resolution while keeping prefix tokens bounded under 360 tokens.
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        w, h = img.size
        if self.max_image_dim and max(w, h) > self.max_image_dim:
            scale = self.max_image_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        return img

    def classify(self, image: Union[str, Path, Image.Image]) -> dict[str, Any]:
        """
        Classifies an image into illustration art vs not art.

        Returns:
            dict containing:
                - is_art: bool
                - label: 'art' or 'not_art'
                - art_probability: float [0.0 - 1.0]
                - confidence: float [0.0 - 1.0]
                - medium: 'illustration_art' | 'screenshot' | 'photograph' | 'diagram'
                - latency_ms: float
                - tokens: int
                - probabilities: dict
        """
        t0 = time.perf_counter()
        prep_img = self.preprocess_image(image)

        state = "Evaluate the visual characteristics and medium of the image."
        res = self.model.system_one(
            state=state,
            questions=CLASSIFIER_QUESTIONS,
            images=[prep_img]
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        ans = res.get("answers", {}).get("medium", {})
        choice = ans.get("choice", "screenshot")
        confidence = float(ans.get("confidence", 0.5))
        probs = ans.get("probabilities", {})

        art_prob = float(probs.get("illustration_art", 0.0))
        is_art = (choice == "illustration_art")
        predicted_label = "art" if is_art else "not_art"

        tokens = int(res.get("usage", {}).get("input_tokens", 0))

        return {
            "is_art": is_art,
            "label": predicted_label,
            "art_probability": round(art_prob, 4),
            "confidence": round(confidence, 4),
            "medium": choice,
            "latency_ms": round(latency_ms, 2),
            "tokens": tokens,
            "probabilities": {k: round(float(v), 4) for k, v in probs.items()},
        }

    # API aliases for evaluation harness compatibility
    def predict_image(self, image: Union[str, Path, Image.Image]) -> dict[str, Any]:
        return self.classify(image)

    def predict(self, image: Union[str, Path, Image.Image]) -> dict[str, Any]:
        return self.classify(image)

    def __call__(self, image: Union[str, Path, Image.Image]) -> dict[str, Any]:
        return self.classify(image)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="D1-Omni Illustration Art Classifier")
    parser.add_argument("image", nargs="?", help="Path to image file to classify")
    parser.add_argument("--device", default="CPU", help="Target execution device")
    args = parser.parse_args()

    classifier = D1ImageClassifier(device=args.device)

    if args.image:
        print(f"\nAnalyzing: {args.image}")
        res = classifier.classify(args.image)
        print(json.dumps(res, indent=2))
    else:
        # Quick self-test using the first manifest items
        manifest_p = Path("dataset_manifest.json")
        if manifest_p.exists():
            items = json.load(manifest_p.open("r", encoding="utf-8"))
            art = next(x for x in items if x["is_art"])
            not_art = next(x for x in items if not x["is_art"])

            print("\n[Self-Test: Art Sample]")
            res1 = classifier.classify(art["path"])
            print(f"  File: {Path(art['path']).name}")
            print(f"  Prediction: {res1['label']} (art_prob={res1['art_probability']*100:.1f}%, latency={res1['latency_ms']:.1f}ms)")

            print("\n[Self-Test: Not-Art Sample]")
            res2 = classifier.classify(not_art["path"])
            print(f"  File: {Path(not_art['path']).name}")
            print(f"  Prediction: {res2['label']} (art_prob={res2['art_probability']*100:.1f}%, latency={res2['latency_ms']:.1f}ms)")
