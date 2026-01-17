from ctok_core.tokenization.io import load_tokenizer, save_tokenizer
from ctok_core.tokenization.rules import RuleSet
from ctok_core.tokenization.vocab import Vocabulary


def test_save_load(tmp_path):
    vocab = Vocabulary(tokens=["a", "b"], special_tokens={})
    rules = RuleSet.from_vocab(vocab)
    save_tokenizer(tmp_path, vocab, rules, metadata={"name": "test"})

    loaded_vocab, loaded_rules, manifest = load_tokenizer(tmp_path)
    assert loaded_vocab.tokens == vocab.tokens
    assert loaded_rules.tokens == rules.tokens
    assert "files" in manifest
