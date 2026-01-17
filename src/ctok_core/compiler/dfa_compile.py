"""Compile vocabulary into a deterministic trie-like DFA."""

from __future__ import annotations

from dataclasses import dataclass

from ctok_core.tokenization.vocab import Vocabulary


@dataclass(frozen=True)
class DFA:
    transitions: list[dict[str, int]]
    terminal_ids: list[int | None]


def compile_dfa(vocab: Vocabulary) -> DFA:
    transitions: list[dict[str, int]] = [{}]
    terminal_ids: list[int | None] = [None]

    for token_id, token in enumerate(vocab.tokens):
        state = 0
        for ch in token:
            next_state = transitions[state].get(ch)
            if next_state is None:
                next_state = len(transitions)
                transitions[state][ch] = next_state
                transitions.append({})
                terminal_ids.append(None)
            state = next_state
        terminal_ids[state] = token_id

    return DFA(transitions=transitions, terminal_ids=terminal_ids)
