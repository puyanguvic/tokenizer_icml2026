from ctok_core.tokenization.runtime import BoundaryAwareTokenizerRuntime, TokenizerRuntime
from ctok_core.tokenization.vocab import Vocabulary


def test_longest_match():
    vocab = Vocabulary(tokens=["a", "ab", "b"], special_tokens={})
    runtime = TokenizerRuntime(vocab)
    ids = runtime.encode("ab")
    tokens = [vocab.token_for(i) for i in ids]
    assert tokens == ["ab"]


def test_fallback_char():
    vocab = Vocabulary(tokens=["a", "b"], special_tokens={})
    runtime = TokenizerRuntime(vocab)
    ids = runtime.encode("ab")
    tokens = [vocab.token_for(i) for i in ids]
    assert tokens == ["a", "b"]


def test_boundary_aware():
    vocab = Vocabulary(tokens=["a", "b", "/"], special_tokens={})
    runtime = BoundaryAwareTokenizerRuntime(vocab, boundary_chars={"/"})
    ids = runtime.encode("a/b")
    tokens = [vocab.token_for(i) for i in ids]
    assert tokens == ["a", "/", "b"]
