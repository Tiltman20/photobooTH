"""Visible traits (glasses, beard, hat, hair colour, shirt colour) with CLIP.

CLIP is a neural network that embeds images and texts into the same space.
Instead of training a classifier per trait we ask *zero-shot questions*: a crop
of a person is compared with a few text prompts per answer ("a person wearing
glasses" vs. "a person without glasses"), and a softmax over the similarities
gives a probability per answer. New traits only need new prompts.

Only the image encoder runs live. Text embeddings are computed once and stored
in ``res/clip_prompt_cache.json`` (part of the repository); the text encoder is
needed only when prompts are added or changed.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from dataclasses import dataclass

import cv2
import numpy as np

import models
from config import ROOT
from face_geometry import FaceFrame, Region

logger = logging.getLogger("photobooth")

PROMPT_CACHE = ROOT / "res" / "clip_prompt_cache.json"
LOGIT_SCALE = 100.0  # CLIP's learned temperature
INPUT_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

# Crops in face coordinates (1 unit = eye distance, origin between the eyes).
CROPS = {
    "head": Region(-1.9, -2.4, 1.9, 1.9),   # hair, hat, glasses and chin
    "body": Region(-2.6, -1.2, 2.6, 5.0),   # head plus upper body: "a person wearing a red shirt"
}
TORSO = Region(-1.8, 2.0, 1.8, 3.4)        # shoulders and chest: must be visible for shirt questions


@dataclass(frozen=True)
class Question:
    """A zero-shot question about one crop; every answer has a few prompts that are averaged."""

    crop: str
    answers: dict[str, tuple[str, ...]]


def _shirt(color: str, *extra: str) -> tuple[str, ...]:
    return (f"a photo of a person wearing a {color} shirt", f"a person in {color} clothes",
            f"a {color} t-shirt", *extra)


QUESTIONS: dict[str, Question] = {
    "glasses": Question("head", {
        "yes": ("a photo of a person wearing glasses", "a close-up of a face with eyeglasses",
                "a person wearing spectacles", "a person wearing sunglasses"),
        "no": ("a photo of a person without glasses", "a close-up of a face with bare eyes",
               "a person not wearing glasses"),
    }),
    "beard": Question("head", {
        "yes": ("a photo of a man with a moustache", "a photo of a man with a beard",
                "a man with a mustache above his lip"),
        "no": ("a photo of a clean-shaven man", "a photo of a young man without a beard",
               "a photo of a woman", "a smooth shaved face"),
    }),
    # Hair styles as counter-answers: "no hat" alone is too vague and gives false alarms.
    "hat": Question("head", {
        "yes": ("a photo of a person wearing a hat", "a person wearing a baseball cap",
                "a person wearing a beanie", "a person wearing a party hat"),
        "short": ("a photo of a person with short hair", "a man with a short haircut"),
        "long": ("a photo of a person with long hair",),
        "curly": ("a photo of a person with curly hair",),
        "bald": ("a photo of a bald person",),
    }),
    "hair": Question("head", {
        "red": ("a photo of a person with red hair", "a redhead with ginger hair"),
        "blond": ("a photo of a person with blond hair", "a person with light blonde hair"),
        "dark": ("a photo of a person with brown hair", "a photo of a person with black hair",
                 "a man with short dark hair", "a person with dark brown hair"),
        "colorful": ("a person with dyed pink hair", "a person with dyed blue hair",
                     "a person with dyed green hair", "a person with dyed purple hair"),
    }),
    "shirt": Question("body", {
        "black": _shirt("black"),
        "white": _shirt("white"),
        "grey": _shirt("grey"),
        "blue": _shirt("blue", "a person wearing a denim jacket"),
        "red": _shirt("red"),
        "green": _shirt("green"),
        "yellow": _shirt("yellow"),
        "pink": _shirt("pink"),
    }),
}


@dataclass(frozen=True)
class Answer:
    """Probabilities of all answers to one question for one person, or why it could not be asked."""

    probabilities: dict[str, float]
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.reason is None

    def best(self) -> tuple[str | None, float]:
        if not self.probabilities:
            return None, 0.0
        return max(self.probabilities.items(), key=lambda item: item[1])


def unusable(reason: str) -> Answer:
    return Answer({}, reason)


# ------------------------------------------------------------------ encoder
class ClipModel:
    """CLIP image encoder (ONNX Runtime) plus cached, averaged text embeddings per answer."""

    def __init__(self) -> None:
        import onnxruntime as ort  # imported here so the rest of the booth works without it

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self._vision = ort.InferenceSession(str(models.ensure(models.CLIP_VISION)), options,
                                            providers=["CPUExecutionProvider"])
        self._lock = threading.Lock()
        self.answer_embeddings = {name: _answer_matrix(question) for name, question in QUESTIONS.items()}

    def embed_images(self, crops: list[np.ndarray]) -> np.ndarray:
        """L2-normalised embeddings of BGR crops (any size)."""
        batch = np.stack([_preprocess(crop) for crop in crops])
        with self._lock:
            embeddings = self._vision.run(["image_embeds"], {"pixel_values": batch})[0].astype(np.float32)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def answer(self, question: str, embedding: np.ndarray) -> Answer:
        names, matrix = self.answer_embeddings[question]
        logits = LOGIT_SCALE * matrix @ embedding
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return Answer({name: float(p) for name, p in zip(names, probabilities)})


def _preprocess(crop: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return ((rgb.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)


def _answer_matrix(question: Question) -> tuple[list[str], np.ndarray]:
    """One row per answer: the normalised mean of its prompt embeddings (prompt ensembling)."""
    names = list(question.answers)
    rows = []
    for name in names:
        mean = text_embeddings(question.answers[name]).mean(axis=0)
        rows.append(mean / np.linalg.norm(mean))
    return names, np.stack(rows)


# ------------------------------------------------------------ text prompts
_cache_lock = threading.Lock()


def text_embeddings(prompts: tuple[str, ...]) -> np.ndarray:
    """Embeddings of the prompts, from the cache file or (for new prompts) from the text encoder."""
    with _cache_lock:
        cache = _load_cache()
        missing = [prompt for prompt in prompts if prompt not in cache]
        if missing:
            logger.info("Berechne %d neue CLIP-Prompts ...", len(missing))
            for prompt, vector in zip(missing, _encode_texts(missing)):
                cache[prompt] = vector
            _save_cache(cache)
    return np.stack([cache[prompt] for prompt in prompts])


def _encode_texts(prompts: list[str]) -> np.ndarray:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(models.ensure(models.CLIP_TOKENIZER)))
    tokenizer.enable_truncation(77)
    tokenizer.enable_padding(length=77, pad_id=49407)  # padding with the end token, as CLIP expects
    ids = np.array([encoding.ids for encoding in tokenizer.encode_batch(prompts)], np.int64)
    session = ort.InferenceSession(str(models.ensure(models.CLIP_TEXT)), providers=["CPUExecutionProvider"])
    embeddings = session.run(["text_embeds"], {"input_ids": ids})[0].astype(np.float32)
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


def _load_cache() -> dict[str, np.ndarray]:
    if not PROMPT_CACHE.is_file():
        return {}
    raw = json.loads(PROMPT_CACHE.read_text(encoding="utf-8"))
    return {prompt: np.frombuffer(base64.b64decode(data), np.float16).astype(np.float32) for prompt, data in raw.items()}


def _save_cache(cache: dict[str, np.ndarray]) -> None:
    raw = {prompt: base64.b64encode(vector.astype(np.float16).tobytes()).decode("ascii")
           for prompt, vector in sorted(cache.items())}
    PROMPT_CACHE.write_text(json.dumps(raw, indent=0), encoding="utf-8")


# ---------------------------------------------------------------- analysis
class AttributeAnalyzer:
    """Answers the questions a challenge needs for every person; the model is loaded on first use."""

    def __init__(self) -> None:
        self._model: ClipModel | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> ClipModel:
        with self._lock:
            if self._model is None:
                logger.info("Lade CLIP-Modell ...")
                self._model = ClipModel()
                logger.info("CLIP-Modell geladen.")
            return self._model

    def analyze(self, frame: np.ndarray, people: list[FaceFrame], questions: set[str],
                min_torso_visible: float) -> list[dict[str, Answer]]:
        """For each person: question -> Answer. One batched network call for all crops."""
        if not questions or not people:
            return [{} for _ in people]
        crop_names = sorted({QUESTIONS[question].crop for question in questions})
        jobs, results = [], [{} for _ in people]
        for index, face_frame in enumerate(people):
            for crop_name in crop_names:
                if crop_name == "body" and face_frame.patch(frame, TORSO, 4).visible_share < min_torso_visible:
                    for question in questions:
                        if QUESTIONS[question].crop == "body":
                            results[index][question] = unusable("Oberkörper nicht im Bild")
                    continue
                region = CROPS[crop_name]
                jobs.append((index, crop_name, crop_patch(frame, face_frame, region)))
        if jobs:
            embeddings = self.model.embed_images([crop for _, _, crop in jobs])
            for (index, crop_name, _), embedding in zip(jobs, embeddings):
                for question in questions:
                    if QUESTIONS[question].crop == crop_name:
                        results[index][question] = self.model.answer(question, embedding)
        return results


def crop_patch(frame: np.ndarray, face_frame: FaceFrame, region: Region) -> np.ndarray:
    """Upright crop of a face region at roughly the network's input resolution."""
    pixels_per_unit = INPUT_SIZE / max(region.x1 - region.x0, region.y1 - region.y0)
    return face_frame.patch(frame, region, pixels_per_unit).image


def all_prompts() -> list[str]:
    return sorted({prompt for question in QUESTIONS.values() for prompts in question.answers.values()
                   for prompt in prompts})


if __name__ == "__main__":
    # After editing prompts: rebuild the cache so that it contains exactly the prompts in use.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PROMPT_CACHE.unlink(missing_ok=True)
    text_embeddings(tuple(all_prompts()))
    print(f"{len(all_prompts())} Prompts in {PROMPT_CACHE.relative_to(ROOT)} gespeichert.")
