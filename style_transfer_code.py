#!/usr/bin/env python3
"""
Final CEFR-preserving literary style-transfer pipeline used in the study.

Procedure
---------
1. Build a target-author style embedding from distributed passages of the
   original (unabridged) novel.
2. Visit sentences of the graded extract in reproducibly shuffled order.
3. Generate several TinyStyler candidates at low style strength.
4. Reject unchanged, malformed, structurally unsafe, overly short/long, or
   semantically dissimilar candidates.
5. Retain only candidates that move toward the original novel's style embedding.
6. Choose among valid candidates probabilistically, using semantic-similarity
   bands and style gain.
7. Reassess the complete extract with CEFR-SP after every tentative replacement.
   Before the minimum number of changes is reached, CEFR-changing candidates are
   rejected and the search continues. After the minimum is reached, the first
   otherwise-valid candidate that would change the aggregate CEFR level is
   rejected and marks the stopping boundary.
8. Save the transferred text and an audit CSV of accepted replacements.

The final outputs should additionally receive a brief manual check for obvious
generation errors (typos/syntax). Stylistically unusual but grammatical wording
should not be normalised merely for sounding unusual.

Required packages:
    torch transformers sentence-transformers huggingface-hub numpy pandas

Inputs are supplied through command-line arguments; no local paths are hard-coded.

Example:
    python style_transfer_code.py \
        --extract graded_extract.txt \
        --original original_novel.txt \
        --cefr-checkpoint level_estimator.ckpt \
        --output transferred_extract.txt

References:
TinyStyler:
    https://aclanthology.org/2024.findings-emnlp.781/
CEFR-SP:
    https://aclanthology.org/2022.emnlp-main.416/
Semantic similarity model:
    https://huggingface.co/sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, set_seed


CEFR_LABELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


# ---------------------------------------------------------------------------
# Text handling
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> list[str]:
    """Split prose while preserving closing quotation marks."""
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return [
        s.strip()
        for s in re.split(
            r'(?<=[.!?])\s+(?=(?:["“‘]?[A-Z]))'
            r'|(?<=[.!?]["”])\s+(?=(?:["“‘]?[A-Z]))',
            text,
        )
        if s.strip()
    ]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_signature(text: str) -> tuple[bool, bool, int]:
    """Record dialogue-boundary structure so generation cannot break quotes."""
    text = text.strip()
    quote_chars = {'"', "“", "”"}
    return (
        bool(text) and text[0] in quote_chars,
        bool(text) and text[-1] in quote_chars,
        sum(ch in quote_chars for ch in text),
    )


def prepare_text_structure(text: str):
    """Keep paragraph/sentence positions so the extract can be reconstructed."""
    paragraphs, locations = [], []

    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        sentences = split_into_sentences(paragraph)
        if not sentences:
            continue

        p = len(paragraphs)
        paragraphs.append(sentences)

        for s, sentence in enumerate(sentences):
            locations.append(
                {"paragraph_index": p, "sentence_index": s, "sentence": sentence}
            )

    return paragraphs, locations


def reconstruct_text(paragraphs: list[list[str]]) -> str:
    return "\n".join(" ".join(p).strip() for p in paragraphs if p)


# ---------------------------------------------------------------------------
# CEFR-SP
# ---------------------------------------------------------------------------

class CEFRSPModel(nn.Module):
    """Inference-only reconstruction of the CEFR-SP level estimator."""

    def __init__(
        self,
        bert_model: str = "bert-base-cased",
        lm_layer: int = 11,
        prototypes_per_level: int = 3,
    ):
        super().__init__()
        self.lm = AutoModel.from_pretrained(bert_model)
        self.lm_layer = lm_layer
        self.prototypes_per_level = prototypes_per_level
        self.prototype = nn.Embedding(
            len(CEFR_LABELS) * prototypes_per_level,
            self.lm.config.hidden_size,
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.lm(**inputs)
        token_embeddings = outputs.hidden_states[self.lm_layer]

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sentence_embeddings = (
            (token_embeddings * mask).sum(dim=1)
            / torch.clamp(mask.sum(dim=1), min=1e-9)
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        prototypes = F.normalize(self.prototype.weight, p=2, dim=1)

        similarities = sentence_embeddings @ prototypes.T
        logits = similarities.reshape(
            -1, self.prototypes_per_level, len(CEFR_LABELS)
        ).mean(dim=1)

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
        return probabilities, predictions


class CEFRSPAssessor:
    """Sentence-level CEFR-SP prediction plus word-weighted text aggregation."""

    def __init__(
        self,
        checkpoint: Path,
        device: str,
        bert_model: str = "bert-base-cased",
    ):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model, use_fast=True)
        self.model = CEFRSPModel(bert_model=bert_model)

        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            saved = torch.load(checkpoint, map_location="cpu")

        state = saved["state_dict"] if "state_dict" in saved else saved
        report = self.model.load_state_dict(state, strict=False)

        allowed = {"lm.embeddings.position_ids", "loss_fct.weight"}
        unexpected = [
            k for k in report.unexpected_keys
            if k not in allowed and not k.endswith("position_ids")
        ]
        if report.missing_keys or unexpected:
            raise RuntimeError(
                f"Incompatible CEFR-SP checkpoint. "
                f"Missing={report.missing_keys}; unexpected={unexpected}"
            )

        self.model.to(self.device).eval()

    def predict(self, sentences: list[str], batch_size: int = 32) -> np.ndarray:
        predictions = []

        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            encoded = self.tokenizer(
                [s.split() for s in batch],
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                _, batch_predictions = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_type_ids=encoded.get("token_type_ids"),
                )

            predictions.extend(batch_predictions.cpu().tolist())

        return np.asarray(predictions)

    def assess(self, text: str) -> dict:
        sentences = split_into_sentences(text)
        if not sentences:
            raise ValueError("No sentences found for CEFR assessment.")

        predictions = self.predict(sentences)
        word_counts = np.asarray([len(s.split()) for s in sentences])
        weighted_mean = float(np.average(predictions, weights=word_counts))

        # Round to nearest CEFR category without Python's banker's rounding.
        aggregate_index = min(
            len(CEFR_LABELS) - 1,
            max(0, math.floor(weighted_mean + 0.5)),
        )

        return {
            "aggregate_cefr": CEFR_LABELS[aggregate_index],
            "weighted_mean": weighted_mean,
        }


# ---------------------------------------------------------------------------
# TinyStyler target style
# ---------------------------------------------------------------------------

def load_tinystyler(repo_id: str, model_name: str, device: str):
    """Load TinyStyler through the authors' Hugging Face interface."""
    interface_file = hf_hub_download(repo_id=repo_id, filename="tinystyler.py")

    spec = importlib.util.spec_from_file_location("tinystyler_module", interface_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    tokenizer, model = module.get_tinystyler_model(
        device=device,
        model_name=model_name,
    )
    model.eval()
    return tokenizer, model, module.get_target_style_embeddings


def make_style_chunks(
    text: str,
    min_words: int,
    target_words: int,
    max_words: int,
) -> list[str]:
    """Create complete-sentence passages suitable for style references."""
    chunks, current, words = [], [], 0

    for sentence in split_into_sentences(text):
        n = len(sentence.split())

        if current and words + n > max_words:
            if words >= min_words:
                chunks.append(" ".join(current))
            current, words = [], 0

        current.append(sentence)
        words += n

        if words >= target_words:
            chunks.append(" ".join(current))
            current, words = [], 0

    if words >= min_words:
        chunks.append(" ".join(current))

    return chunks


def build_target_style(
    original_text: str,
    get_style_embeddings,
    device: str,
    number_of_samples: int,
    min_words: int,
    target_words: int,
    max_words: int,
):
    """
    Represent the original novel using evenly distributed passages.
    TinyStyler combines their authorship embeddings into the target style.
    """
    chunks = make_style_chunks(
        original_text, min_words, target_words, max_words
    )
    if len(chunks) < number_of_samples:
        raise ValueError(
            f"Only {len(chunks)} suitable style chunks were found; "
            f"{number_of_samples} requested."
        )

    positions = np.linspace(
        int(len(chunks) * 0.05),
        int(len(chunks) * 0.95) - 1,
        number_of_samples,
    ).astype(int)

    samples = [chunks[i] for i in positions]
    embedding = get_style_embeddings([samples], device)[0].unsqueeze(0)
    return embedding


# ---------------------------------------------------------------------------
# Candidate generation and filtering
# ---------------------------------------------------------------------------

def similarity_band(score: float) -> str:
    if score >= 0.95:
        return "conservative"
    if score >= 0.91:
        return "moderate"
    return "exploratory"


def candidate_is_well_formed(
    candidate: str,
    source: str,
    min_length_ratio: float,
    max_length_ratio: float,
) -> bool:
    candidate = candidate.strip()

    if not candidate or normalize(candidate) == normalize(source):
        return False
    if any(x in candidate for x in ("<", ">", "_", "http://", "https://")):
        return False
    if re.search(r"(.)\1{5,}", candidate):
        return False
    if candidate[-1] not in '.!?"”’':
        return False
    if quote_signature(candidate) != quote_signature(source):
        return False

    ratio = len(candidate.split()) / max(1, len(source.split()))
    return min_length_ratio <= ratio <= max_length_ratio


def generate_candidate_pool(
    source_sentence: str,
    target_style,
    *,
    tokenizer,
    model,
    get_style_embeddings,
    semantic_model,
    device: str,
    style_strength: float,
    candidates_per_sentence: int,
    min_semantic_similarity: float,
    min_length_ratio: float,
    max_length_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Generate candidates and retain meaning-preserving, style-improving rewrites."""
    source_style = get_style_embeddings([[source_sentence]], device)

    controlled_style = (
        (1.0 - style_strength) * source_style
        + style_strength * target_style
    )

    inputs = tokenizer(
        source_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=192,
    ).to(device)

    output_limit = min(96, max(32, int(inputs["input_ids"].shape[1] * 1.35)))
    set_seed(seed)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            style=controlled_style.to(device),
            do_sample=True,
            temperature=0.8,
            top_p=0.90,
            repetition_penalty=1.10,
            no_repeat_ngram_size=3,
            max_new_tokens=output_limit,
            num_return_sequences=candidates_per_sentence,
        )

    raw = tokenizer.batch_decode(generated, skip_special_tokens=True)

    # De-duplicate, then apply structural filters.
    candidates, seen = [], set()
    for candidate in raw:
        candidate = re.sub(r"\s+", " ", candidate).strip()
        key = normalize(candidate)
        if key not in seen:
            seen.add(key)
            if candidate_is_well_formed(
                candidate, source_sentence, min_length_ratio, max_length_ratio
            ):
                candidates.append(candidate)

    if not candidates:
        return pd.DataFrame()

    # all-mpnet-base-v2 embeddings are normalised; dot product = cosine similarity.
    semantic_vectors = semantic_model.encode(
        [source_sentence] + candidates,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    semantic_scores = (
        semantic_vectors[1:] @ semantic_vectors[0]
    ).detach().cpu().numpy()

    candidate_styles = get_style_embeddings(
        [[candidate] for candidate in candidates], device
    )
    source_target_similarity = F.cosine_similarity(
        source_style, target_style
    ).item()

    rows = []
    for i, candidate in enumerate(candidates):
        semantic_similarity = float(semantic_scores[i])
        target_style_similarity = F.cosine_similarity(
            candidate_styles[i].reshape(1, -1),
            target_style.reshape(1, -1),
        ).item()
        style_gain = target_style_similarity - source_target_similarity

        if semantic_similarity < min_semantic_similarity or style_gain <= 0:
            continue

        rows.append(
            {
                "candidate": candidate,
                "semantic_similarity": semantic_similarity,
                "target_style_similarity": target_style_similarity,
                "style_gain": style_gain,
                "similarity_band": similarity_band(semantic_similarity),
            }
        )

    return pd.DataFrame(rows)


def weighted_candidate_order(
    candidates: pd.DataFrame,
    rng: random.Random,
    band_weights: dict[str, float],
) -> list[dict]:
    """
    Randomise candidate preference while favouring the chosen similarity bands
    and larger movement toward the target style.
    """
    remaining = candidates.copy()
    ordered = []

    while not remaining.empty:
        weights = []
        for _, row in remaining.iterrows():
            style_weight = math.exp(min(row["style_gain"], 0.50) / 0.10)
            weights.append(band_weights[row["similarity_band"]] * style_weight)

        probabilities = np.asarray(weights, dtype=float)
        probabilities /= probabilities.sum()

        position = rng.choices(
            range(len(remaining)), weights=probabilities, k=1
        )[0]
        ordered.append(remaining.iloc[position].to_dict())
        remaining = remaining.drop(
            remaining.index[position]
        ).reset_index(drop=True)

    return ordered


# ---------------------------------------------------------------------------
# Final transfer procedure
# ---------------------------------------------------------------------------

def transfer_until_cefr_boundary(
    source_text: str,
    target_style,
    *,
    cefr_assessor: CEFRSPAssessor,
    tokenizer,
    model,
    get_style_embeddings,
    semantic_model,
    device: str,
    style_strength: float,
    candidates_per_sentence: int,
    min_semantic_similarity: float,
    min_length_ratio: float,
    max_length_ratio: float,
    minimum_changes: int,
    band_weights: dict[str, float],
    random_seed: int,
) -> dict:
    """
    Preserve the baseline whole-text CEFR category.

    Before `minimum_changes`, CEFR-changing candidates are rejected and another
    candidate/sentence is tried. Once the minimum is reached, the first
    otherwise-valid candidate that would cross the CEFR boundary is rejected
    and the procedure stops at the last acceptable version.
    """
    rng = random.Random(random_seed)
    baseline = cefr_assessor.assess(source_text)
    baseline_label = baseline["aggregate_cefr"]

    paragraphs, locations = prepare_text_structure(source_text)
    rng.shuffle(locations)

    accepted, rejected = [], []
    attempts = 0
    boundary = None

    for location in locations:
        p = location["paragraph_index"]
        s = location["sentence_index"]
        source_sentence = paragraphs[p][s]
        attempts += 1

        pool = generate_candidate_pool(
            source_sentence,
            target_style,
            tokenizer=tokenizer,
            model=model,
            get_style_embeddings=get_style_embeddings,
            semantic_model=semantic_model,
            device=device,
            style_strength=style_strength,
            candidates_per_sentence=candidates_per_sentence,
            min_semantic_similarity=min_semantic_similarity,
            min_length_ratio=min_length_ratio,
            max_length_ratio=max_length_ratio,
            seed=random_seed + attempts,
        )

        if pool.empty:
            rejected.append(
                {"source": source_sentence, "reason": "no_safe_style_candidate"}
            )
            continue

        ordered = weighted_candidate_order(pool, rng, band_weights)
        accepted_here = False

        for candidate_data in ordered:
            candidate = candidate_data["candidate"]
            paragraphs[p][s] = candidate

            tentative = cefr_assessor.assess(reconstruct_text(paragraphs))

            if tentative["aggregate_cefr"] == baseline_label:
                accepted.append(
                    {
                        "change_number": len(accepted) + 1,
                        "paragraph_index": p,
                        "sentence_index": s,
                        "source": source_sentence,
                        "replacement": candidate,
                        "semantic_similarity":
                            candidate_data["semantic_similarity"],
                        "target_style_similarity":
                            candidate_data["target_style_similarity"],
                        "style_gain": candidate_data["style_gain"],
                        "similarity_band": candidate_data["similarity_band"],
                        "new_cefr": tentative["aggregate_cefr"],
                        "new_cefr_score": tentative["weighted_mean"],
                    }
                )
                accepted_here = True
                break

            # Revert only the candidate that caused the CEFR change.
            paragraphs[p][s] = source_sentence
            rejected.append(
                {
                    "source": source_sentence,
                    "candidate": candidate,
                    "reason": "CEFR_label_changed",
                    "candidate_cefr": tentative["aggregate_cefr"],
                    "candidate_score": tentative["weighted_mean"],
                }
            )

            if len(accepted) >= minimum_changes:
                boundary = {
                    "source": source_sentence,
                    "candidate": candidate,
                    "resulting_cefr": tentative["aggregate_cefr"],
                    "resulting_score": tentative["weighted_mean"],
                }
                break

        if boundary is not None:
            break

        if not accepted_here:
            paragraphs[p][s] = source_sentence

    final_text = reconstruct_text(paragraphs)
    final_assessment = cefr_assessor.assess(final_text)

    if len(accepted) < minimum_changes:
        stop_reason = "minimum_not_reached"
    elif boundary is not None:
        stop_reason = "next_valid_change_altered_cefr"
    else:
        stop_reason = "all_sentences_examined_without_cefr_change"

    return {
        "text": final_text,
        "baseline": baseline,
        "final_assessment": final_assessment,
        "accepted_changes": pd.DataFrame(accepted),
        "rejected_attempts": pd.DataFrame(rejected),
        "boundary_candidate": boundary,
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CEFR-preserving TinyStyler literary style transfer."
    )
    parser.add_argument("--extract", type=Path, required=True,
                        help="Professionally adapted source extract.")
    parser.add_argument("--original", type=Path, required=True,
                        help="Original unabridged novel used as the target style.")
    parser.add_argument("--cefr-checkpoint", type=Path, required=True,
                        help="Official CEFR-SP level_estimator.ckpt.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination for the transferred text.")
    parser.add_argument("--audit-output", type=Path,
                        help="Optional CSV path for accepted changes.")

    parser.add_argument("--tinystyler-repo", default="tinystyler/tinystyler")
    parser.add_argument("--tinystyler-model", default="tinystyler")
    parser.add_argument("--semantic-model",
                        default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--bert-model", default="bert-base-cased")

    parser.add_argument("--style-strength", type=float, default=0.25)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--min-semantic-similarity", type=float, default=0.88)
    parser.add_argument("--minimum-changes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--style-samples", type=int, default=8)
    parser.add_argument("--style-min-words", type=int, default=80)
    parser.add_argument("--style-target-words", type=int, default=120)
    parser.add_argument("--style-max-words", type=int, default=160)

    parser.add_argument("--min-length-ratio", type=float, default=0.65)
    parser.add_argument("--max-length-ratio", type=float, default=1.45)

    parser.add_argument("--conservative-weight", type=float, default=0.35,
                        help="Weight for similarity >= .95.")
    parser.add_argument("--moderate-weight", type=float, default=0.45,
                        help="Weight for similarity .91-.95.")
    parser.add_argument("--exploratory-weight", type=float, default=0.20,
                        help="Weight for similarity .88-.91.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    source_text = args.extract.read_text(encoding="utf-8")
    original_text = args.original.read_text(encoding="utf-8")

    tokenizer, model, get_style_embeddings = load_tinystyler(
        args.tinystyler_repo, args.tinystyler_model, device
    )
    semantic_model = SentenceTransformer(
        args.semantic_model, device=device
    )
    cefr_assessor = CEFRSPAssessor(
        args.cefr_checkpoint, device, bert_model=args.bert_model
    )

    target_style = build_target_style(
        original_text,
        get_style_embeddings,
        device,
        args.style_samples,
        args.style_min_words,
        args.style_target_words,
        args.style_max_words,
    )

    band_weights = {
        "conservative": args.conservative_weight,
        "moderate": args.moderate_weight,
        "exploratory": args.exploratory_weight,
    }

    result = transfer_until_cefr_boundary(
        source_text,
        target_style,
        cefr_assessor=cefr_assessor,
        tokenizer=tokenizer,
        model=model,
        get_style_embeddings=get_style_embeddings,
        semantic_model=semantic_model,
        device=device,
        style_strength=args.style_strength,
        candidates_per_sentence=args.candidates,
        min_semantic_similarity=args.min_semantic_similarity,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        minimum_changes=args.minimum_changes,
        band_weights=band_weights,
        random_seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result["text"], encoding="utf-8")

    audit_path = args.audit_output or args.output.with_name(
        f"{args.output.stem}_changes.csv"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    result["accepted_changes"].to_csv(
        audit_path, index=False, encoding="utf-8-sig"
    )

    print(f"Baseline CEFR:    {result['baseline']['aggregate_cefr']} "
          f"({result['baseline']['weighted_mean']:.3f})")
    print(f"Final CEFR:       {result['final_assessment']['aggregate_cefr']} "
          f"({result['final_assessment']['weighted_mean']:.3f})")
    print(f"Accepted changes: {len(result['accepted_changes'])}")
    print(f"Stop reason:      {result['stop_reason']}")
    print(f"Text saved to:    {args.output}")
    print(f"Audit saved to:   {audit_path}")


if __name__ == "__main__":
    main()
