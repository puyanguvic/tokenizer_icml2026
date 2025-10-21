from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .tokenizer import DSTTokenizer


DEFAULT_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
TOKEN_SPLIT_RE = re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]")


def _tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text into basic alphanumeric/punctuation units with spans."""
    tokens = []
    for match in TOKEN_SPLIT_RE.finditer(text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens


@dataclasses.dataclass
class ScoreWeights:
    frequency: float = 0.25
    fragmentation: float = 0.2
    compression: float = 0.2
    grammar: float = 0.1
    gradient: float = 0.15
    mutual_information: float = 0.1

    def active_total(self, include_gradient: bool, include_mi: bool) -> float:
        weights = [
            self.frequency,
            self.fragmentation,
            self.compression,
            self.grammar,
            self.gradient if include_gradient else 0.0,
            self.mutual_information if include_mi else 0.0,
        ]
        total = sum(weights)
        return total if total > 0 else 1.0


@dataclasses.dataclass
class CandidateStats:
    token: str
    count: int = 0
    total_fragments: int = 0
    grammar_hits: int = 0
    pmi_sum: float = 0.0
    pmi_count: int = 0

    def record(self, fragments: int, grammar_hit: bool, pmi: float) -> None:
        self.count += 1
        self.total_fragments += fragments
        if grammar_hit:
            self.grammar_hits += 1
        if not math.isnan(pmi):
            self.pmi_sum += pmi
            self.pmi_count += 1

    @property
    def fragmentation(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_fragments / self.count

    @property
    def compression_gain(self) -> float:
        # Each occurrence replaces "fragments" tokens with a single token.
        if self.count == 0:
            return 0.0
        return max(self.fragmentation - 1.0, 0.0) * self.count

    @property
    def grammar_coverage(self) -> float:
        if self.count == 0:
            return 0.0
        return self.grammar_hits / self.count

    @property
    def mutual_information(self) -> float:
        if self.pmi_count == 0:
            return 0.0
        return self.pmi_sum / self.pmi_count


@dataclasses.dataclass
class CandidateExtractorConfig:
    min_frequency: int = 5
    min_length: int = 2
    max_tokens: int = 8
    max_vocab: int = 32000
    special_tokens: Sequence[str] = dataclasses.field(default_factory=lambda: DEFAULT_SPECIAL_TOKENS)
    grammar_patterns: Sequence[str] = dataclasses.field(default_factory=list)
    preserve_case: bool = True
    weights: ScoreWeights = dataclasses.field(default_factory=ScoreWeights)


class CandidateExtractor:
    """Extracts candidate tokens using grammar hints and fragmentation analysis."""

    def __init__(
        self,
        config: CandidateExtractorConfig,
        normalizer: Callable[[str], str],
    ) -> None:
        self.config = config
        self.normalizer = normalizer
        self.grammar_regexes = [re.compile(p) for p in config.grammar_patterns]
        self.unigram_counts: Counter[str] = Counter()
        self.bigram_counts: Counter[Tuple[str, str]] = Counter()
        self.total_unigrams: int = 0

    def _collect_statistics(
        self, corpus: Iterable[str]
    ) -> List[Tuple[str, List[Tuple[str, int, int]]]]:
        processed: List[Tuple[str, List[Tuple[str, int, int]]]] = []
        for raw_line in corpus:
            if not raw_line:
                continue
            line = self.normalizer(raw_line)
            if not self.config.preserve_case:
                line = line.lower()
            spans = _tokenize_with_spans(line)
            token_strings = [tok for tok, _, _ in spans]

            self.unigram_counts.update(token_strings)
            self.bigram_counts.update(zip(token_strings, token_strings[1:]))
            self.total_unigrams += len(token_strings)

            processed.append((line, spans))
        return processed

    @staticmethod
    def _find_token_span(
        spans: Sequence[Tuple[str, int, int]], start: int, end: int
    ) -> Optional[Tuple[int, int]]:
        start_idx: Optional[int] = None
        end_idx: Optional[int] = None
        for idx, (_, tok_start, tok_end) in enumerate(spans):
            if start_idx is None and tok_end > start:
                start_idx = idx
            if tok_start < end:
                end_idx = idx
            if tok_start >= end:
                break
        if start_idx is None or end_idx is None or end_idx < start_idx:
            return None
        return start_idx, end_idx - start_idx + 1

    def _average_pmi(
        self, token_strings: Sequence[str], idx: int, length: int
    ) -> float:
        if length <= 1 or self.total_unigrams == 0:
            return 0.0
        pmis: List[float] = []
        for offset in range(idx, idx + length - 1):
            left = token_strings[offset]
            right = token_strings[offset + 1]
            pair = (left, right)
            pair_count = self.bigram_counts.get(pair, 0)

            if pair_count == 0:
                continue
            p_xy = pair_count / max(self.total_unigrams - 1, 1)
            p_x = self.unigram_counts[left] / self.total_unigrams
            p_y = self.unigram_counts[right] / self.total_unigrams
            denom = p_x * p_y
            if denom <= 0:
                continue
            pmis.append(math.log(p_xy / denom))

        if not pmis:
            return 0.0
        return sum(pmis) / len(pmis)

    def extract(self, corpus: Iterable[str]) -> Dict[str, CandidateStats]:
        processed = self._collect_statistics(corpus)
        candidates: Dict[str, CandidateStats] = {}

        for line, spans in processed:
            token_strings = [tok for tok, _, _ in spans]
            seen_in_line = set()

            for regex in self.grammar_regexes:
                for match in regex.finditer(line):
                    token = match.group(0)
                    span = self._find_token_span(spans, match.start(), match.end())
                    if span is None:
                        continue
                    start_idx, fragments = span
                    if fragments < self.config.min_length:
                        continue
                    pmi = self._average_pmi(token_strings, start_idx, fragments)
                    stats = candidates.setdefault(token, CandidateStats(token=token))
                    stats.record(fragments=fragments, grammar_hit=True, pmi=pmi)
                    seen_in_line.add(token)

            for length in range(2, self.config.max_tokens + 1):
                if length > len(spans):
                    break
                for i in range(0, len(spans) - length + 1):
                    _, start, _ = spans[i]
                    _, _, end = spans[i + length - 1]
                    candidate = line[start:end]
                    if not candidate or candidate.isspace():
                        continue
                    if self.grammar_regexes and candidate in seen_in_line:
                        continue
                    pmi = self._average_pmi(token_strings, i, length)
                    stats = candidates.setdefault(candidate, CandidateStats(token=candidate))
                    stats.record(fragments=length, grammar_hit=False, pmi=pmi)

        filtered = {
            key: stats
            for key, stats in candidates.items()
            if stats.count >= self.config.min_frequency and len(key) >= self.config.min_length
        }
        return filtered


def _normalize_scores(
    candidates: Dict[str, CandidateStats],
    weights: ScoreWeights,
    gradient_scores: Optional[Dict[str, float]] = None,
    mutual_info_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    if not candidates:
        return {}

    max_freq = max(stats.count for stats in candidates.values())
    max_frag = max(stats.fragmentation for stats in candidates.values())
    max_gain = max(stats.compression_gain for stats in candidates.values())
    max_grammar = max(stats.grammar_coverage for stats in candidates.values())

    gradient_scores = gradient_scores or {}
    mutual_info_overrides = mutual_info_overrides or {}

    max_gradient = max(gradient_scores.values(), default=0.0)
    mi_values = [
        mutual_info_overrides.get(tok, stats.mutual_information)
        for tok, stats in candidates.items()
    ]
    max_abs_mi = max((abs(value) for value in mi_values), default=0.0)

    include_gradient = bool(gradient_scores) and weights.gradient > 0
    include_mi = (max_abs_mi > 0 or any(mi_values)) and weights.mutual_information > 0
    active_total = weights.active_total(include_gradient=include_gradient, include_mi=include_mi)

    scores: Dict[str, float] = {}
    for token, stats in candidates.items():
        freq_score = stats.count / max_freq if max_freq else 0.0
        frag_score = stats.fragmentation / max_frag if max_frag else 0.0
        gain_score = stats.compression_gain / max_gain if max_gain else 0.0
        grammar_score = stats.grammar_coverage / max_grammar if max_grammar else 0.0

        grad_score = 0.0
        if include_gradient:
            grad = gradient_scores.get(token, 0.0)
            if max_gradient > 0:
                grad_score = grad / max_gradient

        mi_value = mutual_info_overrides.get(token, stats.mutual_information)
        mi_score = 0.0
        if include_mi and max_abs_mi > 0:
            mi_score = (mi_value / max_abs_mi + 1.0) / 2.0

        weighted = (
            weights.frequency * freq_score
            + weights.fragmentation * frag_score
            + weights.compression * gain_score
            + weights.grammar * grammar_score
        )
        if include_gradient:
            weighted += weights.gradient * grad_score
        if include_mi:
            weighted += weights.mutual_information * mi_score

        scores[token] = weighted / active_total

    return scores


def _build_fallback_tokens() -> List[str]:
    # Include full ASCII range plus whitespace to guarantee coverage.
    fallback = []
    for code in range(32, 127):
        fallback.append(chr(code))
    fallback.extend(["\t", "\n", "\r"])
    return fallback


def build_dst_tokenizer(
    corpus: Iterable[str] | Sequence[str],
    normalizer: Callable[[str], str],
    config: CandidateExtractorConfig,
    save_dir: str | None = None,
    gradient_scores: Optional[Dict[str, float]] = None,
    mutual_info_scores: Optional[Dict[str, float]] = None,
    gradient_provider: Optional[Callable[[Sequence[str], Sequence[str]], Dict[str, float]]] = None,
) -> DSTTokenizer:
    """
    Build a DST tokenizer following the pipeline described in the paper.

    Parameters
    ----------
    corpus:
        Iterable of raw domain strings (already loaded/normalized upstream).
    normalizer:
        Pure function applied before tokenization (e.g., HTTP canonicalization).
    config:
        Candidate extraction configuration (vocabulary budget, grammar hints).
    save_dir:
        Optional directory to serialize artifacts immediately.
    gradient_scores:
        Optional precomputed gradient salience per candidate token.
    mutual_info_scores:
        Optional mutual-information overrides per candidate token.
    gradient_provider:
        Optional callable called as ``gradient_provider(corpus_lines, candidate_tokens)`` to
        compute gradient scores on demand (used if ``gradient_scores`` is not provided).
    """
    corpus_lines = list(corpus)
    normalized_lines = [
        normalizer(line) if config.preserve_case else normalizer(line).lower()
        for line in corpus_lines
    ]

    extractor = CandidateExtractor(config=config, normalizer=normalizer)
    candidates = extractor.extract(corpus_lines)

    if gradient_scores is None and gradient_provider is not None:
        gradient_scores = gradient_provider(normalized_lines, list(candidates.keys()))

    scores = _normalize_scores(
        candidates,
        weights=config.weights,
        gradient_scores=gradient_scores,
        mutual_info_overrides=mutual_info_scores,
    )

    base_tokens = list(config.special_tokens)
    fallback_tokens = _build_fallback_tokens()
    vocab_budget = config.max_vocab - len(base_tokens) - len(fallback_tokens)
    if vocab_budget <= 0:
        vocab_budget = 0

    ranked = sorted(
        ((token, scores.get(token, 0.0), candidates[token]) for token in candidates),
        key=lambda item: item[1],
        reverse=True,
    )

    selected_tokens: List[str] = []
    for token, _, stats in ranked:
        if len(selected_tokens) >= vocab_budget:
            break
        if any(existing in token and existing != token and len(existing) > 1 for existing in selected_tokens):
            continue
        selected_tokens.append(token)

    tokenizer = DSTTokenizer(
        tokens=selected_tokens,
        special_tokens=base_tokens,
        fallback_tokens=fallback_tokens,
        normalizer=normalizer if config.preserve_case else lambda s: normalizer(s).lower(),
        candidate_metadata={
            token: {
                "score": scores.get(token, 0.0),
                "count": candidates[token].count,
                "fragmentation": candidates[token].fragmentation,
                "compression_gain": candidates[token].compression_gain,
                "grammar_coverage": candidates[token].grammar_coverage,
                "mutual_information": candidates[token].mutual_information,
                "gradient_signal": (gradient_scores or {}).get(token),
            }
            for token in selected_tokens
        },
    )

    if save_dir:
        tokenizer.save(save_dir)
    return tokenizer
