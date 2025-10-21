from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence


class GradientSignalProvider:
    """
    Estimate gradient salience for candidate substrings using a masked language model.

    The implementation follows the Stage-II signal described in the DST paper: we
    accumulate gradient norms from an MLM's embedding layer and project them back
    onto character spans, which we then aggregate over candidate substrings.

    Parameters
    ----------
    model:
        A `transformers.PreTrainedModel` supporting masked-language modeling.
    tokenizer:
        The paired `transformers.PreTrainedTokenizerBase`.
    normalizer:
        Optional normalization function applied to raw corpus text before matching.
    device:
        Torch device string (e.g., `"cpu"`, `"cuda"`). Defaults to CPU.
    batch_size:
        Maximum batch size used during gradient computation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        normalizer=None,
        device: str = "cpu",
        batch_size: int = 4,
    ) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:  # pragma: no cover - optional dependency
            raise RuntimeError("PyTorch is required for GradientSignalProvider.") from None

        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.device = device
        self.batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        batch_size: int = 4,
        normalizer=None,
    ) -> "GradientSignalProvider":
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers is required to use GradientSignalProvider.from_pretrained()."
            ) from exc

        model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(
            model=model,
            tokenizer=tokenizer,
            normalizer=normalizer,
            device=device,
            batch_size=batch_size,
        )

    def __call__(
        self, corpus_lines: Sequence[str], candidate_tokens: Sequence[str]
    ) -> Dict[str, float]:
        return self.compute(corpus_lines, candidate_tokens)

    def compute(
        self, corpus_lines: Sequence[str], candidate_tokens: Sequence[str]
    ) -> Dict[str, float]:
        import torch

        candidate_tokens = [token for token in candidate_tokens if token]
        if not candidate_tokens:
            return {}

        # Reduce duplicate candidates while preserving order.
        seen = set()
        unique_candidates: List[str] = []
        for token in candidate_tokens:
            if token not in seen:
                unique_candidates.append(token)
                seen.add(token)

        gradient_scores: Dict[str, float] = defaultdict(float)
        candidate_patterns = {
            token: re.compile(re.escape(token)) for token in unique_candidates
        }

        for start in range(0, len(corpus_lines), self.batch_size):
            batch_texts = list(corpus_lines[start : start + self.batch_size])
            if not batch_texts:
                continue

            if self.normalizer is not None:
                batch_texts = [self.normalizer(text) for text in batch_texts]

            encoding = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
            )

            offsets = encoding.pop("offset_mapping")
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            labels = input_ids.clone()
            embeddings = self.model.get_input_embeddings()(input_ids)
            embeddings.retain_grad()

            outputs = self.model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            token_grads = embeddings.grad.detach().cpu()

            for batch_idx, text in enumerate(batch_texts):
                char_gradients = [0.0] * len(text)
                offset_row = offsets[batch_idx]
                mask_row = (
                    attention_mask[batch_idx].cpu().tolist()
                    if attention_mask is not None
                    else [1] * len(offset_row)
                )

                for pos, grad_vec in enumerate(token_grads[batch_idx]):
                    if mask_row[pos] == 0:
                        continue
                    start_char, end_char = offset_row[pos]
                    if start_char is None or end_char is None:
                        continue
                    start_char = int(start_char)
                    end_char = int(end_char)
                    if start_char == end_char or start_char >= len(char_gradients):
                        continue

                    grad_norm = float(torch.norm(grad_vec, p=2).item())
                    width = max(end_char - start_char, 1)
                    contribution = grad_norm / width
                    for idx_char in range(start_char, min(end_char, len(char_gradients))):
                        char_gradients[idx_char] += contribution

                for token, pattern in candidate_patterns.items():
                    if token not in text:
                        continue
                    for match in pattern.finditer(text):
                        match_start, match_end = match.span()
                        if match_start >= len(char_gradients):
                            continue
                        match_end = min(match_end, len(char_gradients))
                        gradient_scores[token] += sum(
                            char_gradients[match_start:match_end]
                        )

            self.model.zero_grad(set_to_none=True)

        return dict(gradient_scores)


def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """Utility to scale scores into [0, 1] range."""
    if not scores:
        return {}
    max_score = max(scores.values())
    if math.isclose(max_score, 0.0):
        return {key: 0.0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}

