"""Validation utilities for tokenizer artifacts."""

from __future__ import annotations

from ctok.compiler.dfa_compile import compile_dfa
from ctok.tokenization.rules import RuleSet
from ctok.tokenization.vocab import Vocabulary


def validate_vocab(vocab: Vocabulary) -> None:
    _ = compile_dfa(vocab)


def validate_rules(rules: RuleSet) -> None:
    if not rules.tokens:
        raise ValueError("RuleSet has no tokens.")
