"""Validation utilities for tokenizer artifacts."""

from __future__ import annotations

from ctok_core.compiler.dfa_compile import compile_dfa
from ctok_core.tokenization.rules import RuleSet
from ctok_core.tokenization.vocab import Vocabulary


def validate_vocab(vocab: Vocabulary) -> None:
    _ = compile_dfa(vocab)


def validate_rules(rules: RuleSet) -> None:
    if not rules.tokens:
        raise ValueError("RuleSet has no tokens.")
