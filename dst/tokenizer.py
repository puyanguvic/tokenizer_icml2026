import json
from .vocab import GrammarGuidedVocab
from .dfst import DFST


class DSTTokenizer:
    """Main interface for Domain-Specific Tokenization."""

    def __init__(self, vocab, dfst):
        self.vocab = vocab
        self.dfst = dfst

    @classmethod
    def train(cls, corpus, **kwargs):
        gg = GrammarGuidedVocab(**kwargs)
        vocab = gg.build_vocab(corpus)
        dfst = DFST()
        for token in vocab:
            dfst.add_token(token)
        return cls(vocab, dfst)

    def encode(self, text: str):
        return self.dfst.encode(text)

    def decode(self, tokens):
        return self.dfst.decode(tokens)

    def verify(self, corpus):
        return self.dfst.verify(corpus)

    def save_json(self, path="dst_tokenizer.json"):
        """Export tokenizer to Hugging Face-compatible JSON format."""
        obj = {
            "version": "1.0",
            "vocab": {t: i for i, t in enumerate(self.vocab)},
            "normalizer": {"type": "identity"},
            "pre_tokenizer": {"type": "Whitespace"},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_vocab(cls, vocab: list[str]):
        """Construct a DSTTokenizer directly from an explicit vocabulary.

        This bypasses grammar-guided induction and simply builds the DFST
        trie from the provided tokens.
        """
        dfst = DFST()
        for token in vocab:
            dfst.add_token(token)
        return cls(vocab, dfst)

    @classmethod
    def load_json(cls, path: str):
        """Load a serialized tokenizer JSON and rebuild DFST.

        Expects the format produced by `save_json`, where `vocab` is a
        mapping of token -> id. Tokens are restored in id order.
        """
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        vocab_map = obj.get("vocab", {})
        # Restore tokens in id order for stable indexing
        tokens_sorted = [t for t, _ in sorted(vocab_map.items(), key=lambda kv: kv[1])]
        return cls.from_vocab(tokens_sorted)
